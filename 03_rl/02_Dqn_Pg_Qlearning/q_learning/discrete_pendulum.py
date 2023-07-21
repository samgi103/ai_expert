import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from typing import List
from gym.utils.renderer import Renderer


class DiscretePendulumEnv(gym.Env):
    # forked from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
    # modified so that the environment eventually becomes a discrete MDP with stochastic transition
    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, g=10.0, n_th=40, n_thdot=60, num_u=15, render_mode=None):
        # system parameters
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.L = 1.
        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        # note that (# of points along \dot\theta-axis) is (# of cells + 1)!
        # however, (# of points along \theta-axis) = (# of cells)
        self.n_th = n_th        # number of cells along \theta-axis
        self.n_thdot = n_thdot     # number of cells along \dot\theta-axis

        self.n_pts = self.n_th * (self.n_thdot + 1)
        self.width_th = 2. * np.pi / self.n_th
        self.width_thdot = 2. * self.max_speed / self.n_thdot
        self.num_u = num_u
        self.u_interval = 2. * self.max_torque / (self.num_u - 1)
        self.action_space = spaces.Discrete(self.num_u)
        self.observation_space = spaces.Discrete(self.n_pts)

        self.x = None
        self.seed()

    def step(self, action: int):
        assert action < self.action_space.n
        u = -self.max_torque + action * self.u_interval     # discrete to continuous
        costs = self.cost(self.x, u)
        # s_{t+1} ~ p(s_t, a_t)
        x_next = self.f(x=self.x, u=u)
        discrete_state = self.to_discrete(x_next)
        self.x = self.to_continuous(discrete_state)
        self.renderer.render_step()

        return discrete_state, -costs, False, {}

    def reset(self, deterministic=False):
        if deterministic:
            discrete_state = 30
        else:
            high = np.array([np.pi, 1])
            
            discrete_state = self.to_discrete(self.np_random.uniform(low=-high, high=high))
        self.x = self.to_continuous(discrete_state)
        self.state = np.copy(self.x)
        self.last_u = None
        self.renderer.reset()
        self.renderer.render_step()

        return discrete_state

    def f(self, x, u):
        # simulate pendulum dynamics in continuous state space
        th, thdot = x
        g = self.g
        m = self.m
        L = self.L
        dt = self.dt
        newthdot = thdot + (-3. * g / (2. * L) * np.sin(th + np.pi) + 3. / (m * L ** 2) * u) * dt
        newth = angle_normalize(th + newthdot * dt)
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        x_next = np.array([newth, newthdot])
        self.state = np.copy(x_next)
        return x_next

    @staticmethod
    def cost(x, u):
        # quadratic cost ftn
        th, thdot = x
        return th ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

    def to_continuous(self, d) -> np.ndarray:
        n1, n2 = d // (self.n_thdot + 1), d % (self.n_thdot + 1)
        x1, x2 = -np.pi + n1 * self.width_th, -self.max_speed + n2 * self.width_thdot
        return np.array([x1, x2])

    def to_discrete(self, x) -> int:
        # sample a discrete state corresponding to the current continuous state
        # computed under induced stochastic transition
        p = self.bin(x)
        d = np.random.choice(self.n_pts, p=p)
        return d

    def bin(self, x: np.ndarray) -> List[float]:
        # binning continuous state variable using Kuhn triangulation
        # easily generalizes to high-dimensional case
        # return a probability vector $p$, where $p(i)$ indicates the probability of $x$ being approximated by a pt $i$
        th, thdot = x
        n1, x1 = int((th + np.pi) // self.width_th), (th + np.pi) % self.width_th
        n2, x2 = int((thdot + self.max_speed) // self.width_thdot), (thdot + self.max_speed) % self.width_thdot
        # normalize both x1 and x2 so that $x = (x1, x2)^\top$ belongs to the unit cube $[0, 1)^2$
        x1 /= self.width_th
        x2 /= self.width_thdot
        d = n1 * (self.n_thdot + 1) + n2
        # subdivision of a cube into two simplices (n! in general)
        # mesh arranged in lexicographic manner as follows:
        # $x_1 > x_2$ iff $\theta_1 > theta_2$ or $\theta_1 = \theta_2, \dot\theta_1 > \dot\theta_2$
        p = [0.] * self.n_pts
        if x1 >= x2:
            # since the original state space is cylindrical, take mod N, where N = (size of the state space)
            d1, d2, d3 = d, (d + self.n_thdot + 1) % self.n_pts, (d + 2 + self.n_thdot) % self.n_pts
            p[d1], p[d2], p[d3] = 1. - x1, x1 - x2, x2
        else:
            d1, d2, d3 = d, (d + 1) % self.n_pts, (d + 2 + self.n_thdot) % self.n_pts
            p[d1], p[d2], p[d3] = 1 - x2, x2 - x1, x1
        return p

    def render(self, mode="human"):
        if self.render_mode is not None:
            return self.renderer.get_renders()
        else:
            return self._render(mode)

    def _render(self, mode="human"):
        assert mode in self.metadata["render_modes"]
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in {"rgb_array", "single_rgb_array"}
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if self.last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2),
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


def angle_normalize(theta):
    # $[-pi, pi)$
    return ((theta + np.pi) % (2. * np.pi)) - np.pi

