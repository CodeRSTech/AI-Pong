from xml.etree import ElementTree

from scripts.generate_diagrams import (
    build_crossover_svg,
    build_network_svg,
    generate_diagrams,
)
from src.ga.player import IndividualPlayer


SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"


def test_network_diagram_uses_current_architecture_and_full_connectivity():
    svg = ElementTree.fromstring(build_network_svg(IndividualPlayer()))

    assert svg.find(f"{SVG_NAMESPACE}title").text == "AI-Pong neural network"
    assert len(svg.findall(f"{SVG_NAMESPACE}circle")) == 23
    assert len(svg.findall(f"{SVG_NAMESPACE}line")) == 116
    text = " ".join(node.text or "" for node in svg.findall(f"{SVG_NAMESPACE}text"))
    assert "Observations (7)" in text
    assert "Layer 1 (8)" in text
    assert "Layer 2 (6)" in text
    assert "Actions (2)" in text


def test_crossover_diagram_shows_reciprocal_children_and_weight_origins():
    diagram = build_crossover_svg(IndividualPlayer(), (3, 12))
    svg = ElementTree.fromstring(diagram)

    text = " ".join(node.text or "" for node in svg.findall(f"{SVG_NAMESPACE}text"))
    assert "Child A" in text
    assert "Child B" in text
    assert "cut 1" in text
    assert "cut 2" in text
    assert len(svg.findall(f"{SVG_NAMESPACE}line")) == 236
    assert "{offset + local_index}" not in diagram


def test_diagram_generator_writes_deterministic_svg_files(tmp_path):
    network_path, crossover_path = generate_diagrams(tmp_path)
    first_outputs = (network_path.read_bytes(), crossover_path.read_bytes())

    generate_diagrams(tmp_path)

    assert first_outputs == (network_path.read_bytes(), crossover_path.read_bytes())
    assert network_path.name == "neural-network.svg"
    assert crossover_path.name == "two-point-crossover.svg"
