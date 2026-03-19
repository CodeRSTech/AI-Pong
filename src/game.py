import pygame as pg
from playzone import PlayZone
from variables import *


class Game:
    """
    The game class that manages the game loop and player interactions.
    """

    def __init__(self, players, width=VARIABLES['WIDTH'], height=VARIABLES['HEIGHT'],
                 fps=VARIABLES['FPS'], timeout=VARIABLES['TIME_OUT'],
                 speed=VARIABLES['SPEED']):
        """
        Initialize the game with the specified parameters.

        Args:
            players (list): List of players (IndividualPlayer instances).
            width (int): Width of the display in pixels.
            height (int): Height of the display in pixels.
            fps (int): Frames per second.
            timeout (float): Amount of time for which the game will run.
            speed (float): Speed of the game.
        """
        # Initialise pygame, font, game attributes, clock object, screens and play-zones
        pg.init()
        pg.display.set_caption("Pong 2D")
        self.font = pg.font.SysFont("couriernew", 22)
        self.display_score = False
        self.fps = fps
        self.speed = speed
        self.timeout = timeout
        self.players = players
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((width, height))
        self.playzone_surface = pg.Surface((width, height))
        self.num_zones = len(players)
        self.zones = []

        # Generate Play-Zones
        best_score = players[0].scores['overall_fitness']
        for i in range(self.num_zones):
            zone = PlayZone(width, height, speed, players[i], best_score)
            self.zones.append(zone)

    def start(self) -> list:
        """
        Run the game and return the updated population.

        Returns:
        list: List of IndividualPlayer instances.
        """
        running = True
        time_running = 0

        # Begin Game Loop
        while running and (time_running <= self.timeout or self.timeout == -1):

            # Process events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

            # Fill the surface with color
            self.playzone_surface.fill(OFF_WHITE)

            # Update each Play-Zone and render to `playzone_surface`
            for container in self.zones:
                container.update()
                container.render_to(self.playzone_surface)

            # Render `playzone_surface` to the screen
            self.screen.blit(self.playzone_surface, (0, 0))

            # Display scores (used by tester.py)
            if self.num_zones == 1 and self.display_score:
                scores = self.players[0].scores
                score_text = "Player:{0} CPU:{1}".format(scores['player'],scores['cpu'])
                text_surface = self.font.render(score_text, True, GRAY)
                pos_x = self.screen.get_width() // 2 - text_surface.get_width() // 2
                pos_y = self.screen.get_height() // 2
                self.screen.blit(text_surface, (pos_x, pos_y))

            # Display
            pg.display.flip()

            # Update the timer
            self.clock.tick(self.fps * self.speed)
            time_running += 1 / (self.speed * self.fps)
        return self.players
