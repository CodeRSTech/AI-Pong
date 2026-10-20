"""Generate virtual API pages and navigation without importing the application."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import posixpath
import re

from mkdocs.exceptions import PluginError
from mkdocs.structure.files import File, InclusionLevel


API_ROOT = PurePosixPath("reference/api")
API_ASSET_ROOT = PurePosixPath("assets/api")
API_ASSET_REFERENCE = re.compile(r'(?P<prefix>(?<![\w:-])src=["\'])api-asset:(?P<path>[^"\']+)(?P<quote>["\'])')
API_NAV_TITLE = "API Reference"
# [NOTE] Key overrides by defining module so private APIs stay opt-in and ownership is clear.
MEMBER_EXCEPTIONS: dict[str, tuple[str, ...]] = {}
MEMBER_EXCLUSIONS: dict[str, tuple[str, ...]] = {}


def is_type_expression(value: ast.expr | None) -> bool:
    """Recognize conventional assignment-style aliases without evaluating code."""
    if isinstance(value, ast.Subscript):
        return is_type_expression(value.value)
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.BitOr):
        return is_type_expression(value.left) and is_type_expression(value.right)
    if isinstance(value, ast.Name):
        return value.id[:1].isupper() or value.id in {
            "list",
            "tuple",
            "dict",
            "set",
            "frozenset",
            "str",
            "int",
            "float",
            "bool",
            "bytes",
            "object",
        }
    if isinstance(value, ast.Attribute):
        return value.attr[:1].isupper()
    return isinstance(value, ast.Constant) and value.value is None


def is_api_assignment(name: str, value: ast.expr | None, annotation: ast.expr | None = None) -> bool:
    """Decide whether a module-level assignment is part of the documented API."""
    if name.startswith("_") or name in {"logger", "log"}:
        return False
    if name.isupper():
        return True
    if isinstance(annotation, ast.Name) and annotation.id == "TypeAlias":
        return True
    if isinstance(annotation, ast.Attribute) and annotation.attr == "TypeAlias":
        return True
    return name[:1].isupper() and not isinstance(value, ast.Constant) and is_type_expression(value)


@dataclass(frozen=True)
class Module:
    """Describe a Python source module and its generated API page."""

    name: str
    source: Path
    page: PurePosixPath
    is_package: bool


def discover_modules(root: Path) -> list[Module]:
    """Map each Python file under the `src` package to a stable virtual page."""
    modules = []
    pages: set[PurePosixPath] = set()
    for source in sorted((root / "src").rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        parts = source.relative_to(root).with_suffix("").parts
        is_package = parts[-1] == "__init__"
        name_parts = parts[:-1] if is_package else parts
        page = API_ROOT.joinpath(*name_parts, "index.md")
        if page in pages:
            raise PluginError(f"API page collision: {page} ({source})")
        pages.add(page)
        modules.append(Module(".".join(name_parts), source, page, is_package))
    if not modules:
        raise PluginError(f"No Python modules found in {root / 'src'}")
    return modules


def local_members(module: Module) -> list[str]:
    """Collect defining-module members, never imported aliases or external APIs."""
    tree = ast.parse(module.source.read_text(encoding="utf-8"), filename=str(module.source))
    names: list[str] = []

    def collect(statements: list[ast.stmt]) -> None:
        """Collect declarations in module scope, including conditional declarations."""
        for node in statements:
            candidates: list[str] = []
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                candidates = [node.name]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                candidates = [node.target.id] if is_api_assignment(node.target.id, node.value, node.annotation) else []
            elif hasattr(ast, "TypeAlias") and isinstance(node, ast.TypeAlias):
                candidates = [node.name.id]
            elif isinstance(node, ast.Assign):
                candidates = [
                    target.id
                    for target in node.targets
                    if isinstance(target, ast.Name) and is_api_assignment(target.id, node.value)
                ]
            elif isinstance(node, ast.If):
                # [NOTE] Public declarations may be guarded by platform or optional-dependency branches.
                collect(node.body)
                collect(node.orelse)
            elif isinstance(node, (ast.Try,) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())):
                collect(node.body)
                collect(node.orelse)
                collect(node.finalbody)
                for handler in node.handlers:
                    collect(handler.body)
            names.extend(name for name in candidates if not name.startswith("_"))

    collect(tree.body)
    names.extend(MEMBER_EXCEPTIONS.get(module.name, ()))
    excluded = set(MEMBER_EXCLUSIONS.get(module.name, ()))
    return [name for name in dict.fromkeys(names) if name not in excluded]


def relative_link(origin: PurePosixPath, target: PurePosixPath) -> str:
    """Build a relative URL from one generated page to another."""
    return posixpath.relpath(str(target), str(origin.parent))


def children(module: Module, modules: list[Module]) -> list[Module]:
    """Return modules nested immediately under the given package."""
    return [child for child in modules if child.name.rpartition(".")[0] == module.name]


def resolve_export(
    module: Module,
    symbol: str,
    modules: dict[str, Module],
    seen: set[tuple[str, str]],
) -> Module | None:
    """Follow internal re-export chains to the defining module, without imports."""
    key = (module.name, symbol)
    # [NOTE] Stop cycles in static package re-exports instead of resolving imports at runtime.
    if key in seen:
        return None
    seen.add(key)
    if symbol in local_members(module):
        return module
    tree = ast.parse(module.source.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        base = module.name.split(".") if module.is_package else module.name.split(".")[:-1]
        if node.level:
            base = base[: len(base) - node.level + 1]
            imported = ".".join(base + ([node.module] if node.module else []))
        else:
            imported = node.module or ""
        for alias in node.names:
            if (alias.asname or alias.name) != symbol:
                continue
            submodule = modules.get(f"{imported}.{alias.name}")
            if submodule is not None:
                return submodule
            target = modules.get(imported)
            if target is not None:
                return resolve_export(target, alias.name, modules, seen)
    return None


def module_page(module: Module, modules: list[Module], root: Path, repo_url: str) -> str:
    """Compose a virtual Markdown page for one module or package."""
    source = module.source.relative_to(root).as_posix()
    title = module.name.rsplit(".", 1)[-1].replace("_", " ").title()
    text = (
        f'# {title}\n\n<div class="api-module-path"><code>{module.name}</code></div>\n'
        f"[View source code]({repo_url}/blob/main/{source})\n"
    )
    members = local_members(module)
    text += f"::: {module.name}\n    options:\n        members: {members!r}\n"
    if module.is_package:
        text += "\n## Modules and packages\n\n"
        for child in children(module, modules):
            text += f"- [`{child.name}`]({relative_link(module.page, child.page)})\n"
        # [NOTE] Link aliases to their defining page; rendering them here would duplicate symbol anchors.
        tree = ast.parse(module.source.read_text(encoding="utf-8"))
        exports = next(
            (
                ast.literal_eval(node.value)
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
                and isinstance(node.value, (ast.List, ast.Tuple))
            ),
            (),
        )
        targets = {item.name: item for item in modules}
        links = []
        for public in exports:
            target = resolve_export(module, public, targets, set())
            if target is not None and target != module:
                links.append(f"- `{public}`: [{target.name}]({relative_link(module.page, target.page)})\n")
        if links:
            text += "\n## Public re-exports\n\n" + "".join(links)
    return text


def api_navigation(modules: list[Module], parent: str = "src") -> list:
    """Build API navigation while keeping the top-level `src` package out of the menu."""
    module = next(item for item in modules if item.name == parent)
    result: list = [] if parent == "src" else [str(module.page)]
    for child in children(module, modules):
        label = child.name.rsplit(".", 1)[-1]
        descendants = children(child, modules)
        result.append(
            {label: api_navigation(modules, child.name) if child.is_package and descendants else str(child.page)}
        )
    return result


def on_files(files, config):
    """Prepare MkDocs files and navigation for the generated reference.

    API Markdown stays in the build's file collection; generated pages are not
    written into the documentation source tree.
    """
    root = Path(config.config_file_path).resolve().parent
    modules = discover_modules(root)
    for file in list(files):
        page = PurePosixPath(file.src_uri)
        if page.is_relative_to(API_ROOT):
            raise PluginError(f"{API_ROOT} is reserved for generated pages: {page}")
    for module in modules:
        files.append(
            File.generated(
                config,
                str(module.page),
                content=module_page(module, modules, root, config.repo_url.rstrip("/")),
            )
        )
    landing = "# API reference\n\nGenerated from source; edit Python docstrings, not these pages.\n\n"
    package = next(module for module in modules if module.name == "src")
    landing += f"[src package index]({relative_link(API_ROOT / 'index.md', package.page)})\n\n"
    for module in children(package, modules):
        landing += f"- [{module.name.rsplit('.', 1)[-1]}]({relative_link(API_ROOT / 'index.md', module.page)})\n"
    files.append(File.generated(config, str(API_ROOT / "index.md"), content=landing))
    api = next((entry for entry in config.nav if API_NAV_TITLE in entry), None)
    if api is None:
        raise PluginError(f"mkdocs.yml must contain an {API_NAV_TITLE!r} navigation entry")
    # [NOTE] Replace navigation on every rebuild because `mkdocs serve` reuses this config object.
    api[API_NAV_TITLE] = [str(API_ROOT / "index.md"), *api_navigation(modules)]
    # [NOTE] Keep the source package reachable from the landing page without adding it to the menu.
    files.get_file_from_path(str(package.page)).inclusion = InclusionLevel.NOT_IN_NAV
    return files


def on_page_context(context, page, config, nav):
    """Remove edit links from generated pages that have no source Markdown file."""
    if PurePosixPath(page.file.src_uri).is_relative_to(API_ROOT):
        page.edit_url = None
    return context


def on_post_page(output, page, config):
    """Resolve stable API-asset references after docstrings have been rendered."""
    page_path = PurePosixPath(page.file.src_uri)
    if not page_path.is_relative_to(API_ROOT):
        return output

    docs_dir = Path(config.docs_dir).resolve()
    asset_root = (docs_dir / Path(*API_ASSET_ROOT.parts)).resolve()

    def resolve_asset(match: re.Match[str]) -> str:
        """Validate an asset reference and replace it with a page-relative URL."""
        raw_path = match.group("path")
        asset = PurePosixPath(raw_path)
        # [NOTE] Reject traversal and absolute paths before resolving under the dedicated asset root.
        if "\\" in raw_path or asset.is_absolute() or not asset.parts or ".." in asset.parts:
            raise PluginError(f"Invalid API asset reference: {raw_path}")

        target = API_ASSET_ROOT / asset
        asset_path = (docs_dir / Path(*target.parts)).resolve()
        if not asset_path.is_relative_to(asset_root) or not asset_path.is_file():
            raise PluginError(f"API asset does not exist: {target}")

        return f"{match.group('prefix')}{relative_link(page_path, target)}{match.group('quote')}"

    return API_ASSET_REFERENCE.sub(resolve_asset, output)


def on_serve(server, config, builder):
    """Watch the src package so API pages rebuild during live preview."""
    root = Path(config.config_file_path).resolve().parent
    server.watch(str(root / "src"), builder)
    return server
