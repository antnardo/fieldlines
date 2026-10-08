#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 14:07:22 2017

Stream Lines codes for Electrostatic field of discrete charges

e/4*pi*epsilon_0=1

@author: Antonin Marchand

See examples

Sources :
    http://nbviewer.jupyter.org/url/elektromagnetisme.no/wp-content/uploads/2013/09/quadpole_preview.ipynb
    http://tonysyu.github.io/plotting-streamlines-with-matplotlib-and-sympy.html
    http://numbercrunch.de/blog/2013/05/visualizing-streamlines/
    https://commons.wikimedia.org/wiki/User:Geek3/VectorFieldPlot
"""

# import useful modules
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from scipy.integrate import ode
import numpy as np


def getkwargs(dic, key, defaultvalue):
    v = dic.get(key)
    if v is None:
        v = defaultvalue
    return v


class Source:
    """ Classe générale d'accès aux sources
    les calculs de V et E doivent être vectorisables
    """
    def __init__(self, q, pos) -> None:
        """q=float : intensité, pos=(x,y)"""
        self.q = q
        self.pos = pos  # (x, y)
        self.x = self.pos[0]
        self.y = self.pos[1]

    def _r(self, x, y):
        x0, y0 = self.pos
        return ((x - x0) ** 2 + (y - y0) ** 2) ** (.5)

    def E(self, x, y):
        NotImplemented

    def V(self, x, y):
        NotImplemented


class Charge(Source):
    """ Classe qui contient les propriétés d'une charge ou d'un dipole
    ainsi que le champ et le potentiel qu'il crée

    Charge(q, pos, source='charge', alpha=0)
    q:intensité (flottant)
    pos = (x, y)
    source = 'charge' (default) ou 'dipole'
    alpha = (default 0) angle du dipole
    """
    def __init__(self, q, pos):
        super().__init__(q, pos)

    def E(self, x, y):
        q = self.q
        r3 = self._r(x, y)**3
        x0, y0 = self.pos
        return (
            q * (x - x0) / r3,
            q * (y - y0) / r3,
        )

    def V(self, x, y):
        return self.q / self._r(x, y)

class Ligne(Source):
    """ Ligne de charge lambda -> q = 2 lambda"""
    def __init__(self, q, pos):
        super().__init__(q, pos)

    def E(self, x, y):
        q = self.q
        x0, y0 = self.pos
        r2 = self._r(x, y)**2
        return (
            q * (x - x0) / r2,
            q * (y - y0) / r2,
        )

    def V(self, x, y):
        return - self.q * np.log(self._r(x, y))


class Dipole(Source):
    def __init__(self, q, pos, alpha=0) -> None:
        """ 
        alpha = (default 0) angle du dipole
        """
        super().__init__(q, pos)
        self.alpha = alpha

    def E(self, x, y):
        x0, y0 = self.pos
        p = self.q
        r3 = self._r(x, y)**3
        theta = np.arctan2(y - y0, x - x0)
        thetad = np.pi / 2 - theta - self.alpha
        Er = 2 * p * np.cos(thetad) / r3
        Etheta = p * np.sin(thetad) / r3
        Ex = Er * np.cos(theta) + Etheta * np.sin(theta)
        Ey = Er * np.sin(theta) - Etheta * np.cos(theta)
        return Ex, Ey

    def V(self, x, y):
        x0, y0 = self.pos
        p = self.q
        r = self._r(x, y)
        theta = np.arctan2(y - y0, x - x0)
        thetad = np.pi / 2 - theta - self.alpha
        return p * np.cos(thetad) / r ** 2


class Streamline:
    """ Classe qui contient les propriétés d'une ligne de champ et de son calcul pour une figure donnée

    Streamline(x0, y0, dt=0.01, R=0.2, name='', enlarge = 1000, arrow_positions=[0.5],verbose=True, Emax = 1000, Emin = 0.01

    x0, y0:position de départ
    dt:pas d'intégration
    R:rayon de sécurité autour du point de départ pour détecter les lignes de champ qui stagnent au départ
    enlarge:coefficient qui permet de ne pas stopper tout de suite les intégrations pour des lignes de champ qui sortent de la figure
    arrow_positions:positions le long de la ligne des flèches, bornées par la figure
    Emax:détection des lignes qui divergent
    Emin:idem

    méthodes utiles :
        integrate_streamlines(int_functiondir, int_functionnorm, figure_limits)
        add_arrows(positions=[], **kwargs)
    """

    def __init__(
        self,
        x0,
        y0,
        dt=0.01,
        R=0.2,
        name="",
        enlarge=1,
        arrow_positions=[0.5],
        verbose=True,
        Emax=1000,
        Emin=0.0001,
    ):
        self.x = [x0]  # start position
        self.y = [y0]  # start position
        self.dt = dt  # integration step
        self.t = dt  # next integration time
        self.tl = [0]  # time at x and y
        self.R = (
            R  # safety radius around init position, from which integration must escape
        )
        self.backw = False
        self.backwindex = 0
        self.verbose = verbose
        self.arrow_positions = arrow_positions
        self.name = name
        self.enlarge = enlarge
        self.Emax = Emax
        self.Emin = Emin
        self.patches = []

    def _log(self, s):
        if self.verbose:
            print("SL{:s}: ".format(self.name) + s)

    def change_parameter(
        self,
        x0=None,
        y0=None,
        dt=None,
        R=None,
        name=None,
        enlarge=None,
        arrow_positions=None,
        Emax=None,
        Emin=None,
    ):
        if enlarge:
            self.enlarge = enlarge
        if Emax:
            self.Emax = Emax
        if Emin:
            self.Emin = Emin
        if name:
            self.name = name
        if R:
            self.R = R
        if dt:
            self.dt = dt
        if arrow_positions:
            self.arrow_positions = arrow_positions
        if x0:
            self.x = [x0]  # start position
        if y0:
            self.y = [y0]  # start position

    @property
    def len(self):
        return len(self.x)

    def addpoint(self, x, y):
        self.x.append(x)
        self.y.append(y)
        t2 = self.t + self.dt
        self.t = t2
        self.tl.append(t2)

    def backwards(self):
        self.x.append(self.x[0])
        self.y.append(self.y[0])
        dt = -self.dt
        self.dt = dt
        self.t = dt
        self.tl.append(0)
        self.backw = True
        self.backwindex = self.len

    def sort(self):
        if self.backw:
            N = self.len
            self.x = (self.x[N:self.backwindex - 1:-1] + self.x[0:self.backwindex - 2])
            self.y = (self.y[N:self.backwindex - 1:-1] + self.y[0:self.backwindex - 2])
            self.tl = (self.y[N:self.backwindex - 1:-1] + self.tl[0:self.backwindex - 2])

    # Compute where integration can escape from limits
    def enlargemin(self, x, coef=None):
        if coef is None:
            coef = self.enlarge
        if x < 0:
            return coef * x
        else:
            return x / coef

    def enlargemax(self, x):
        coef = self.enlarge
        return self.enlargemin(x, 1 / coef)

    def integrate_streamlines(
        self, int_functiondir, int_functionnorm, figure_limits, int_param=None
    ):
        """Main integration algorithm """
        xmin, xmax, ymin, ymax = figure_limits
        r = ode(int_functiondir)
        r.set_integrator("vode")
        if int_param is not None:
            r.set_f_params(int_param)
        r.set_initial_value([self.x[0], self.y[0]], self.tl[0])
        hit_edges = False
        while r.successful():
            r.integrate(self.t)
            x = r.y[0]
            y = r.y[1]
            self.addpoint(x, y)
            # stop conditions
            # out of plot, if first edge encountered, go backwards
            E = int_functionnorm(x, y, int_param)
            if (not (x > self.enlargemin(xmin) and x < self.enlargemax(xmax))) or (
                not (y > self.enlargemin(ymin) and y < self.enlargemax(ymax))
            ):
                if hit_edges:
                    self._log("second edge stop at t={:.3e}".format(self.t))
                    break
                else:
                    hit_edges = True
                    self._log(
                        "first edge at t={:.3e}, going backwards, index={:d}".format(
                            self.t, self.len - 1
                        )
                    )
                    self.backwards()
                    r.set_initial_value([self.x[-1], self.y[-1]], self.tl[-1])
            # E divergent, go backwards
            elif E > self.Emax:
                if hit_edges:
                    self._log(
                        "infinity field stop encountered at t={:.3e}, x={:.3e}, y={:.3e}, E={:3e}".format(
                            self.t, x, y, E
                        )
                    )
                    break
                else:
                    self._log(
                        "infinity field encountered at t={:.3e}, x={:.3e}, y={:.3e}, E={:3e}, going backwards".format(
                            self.t, x, y, E
                        )
                    )
                    self.backwards()
                    r.set_initial_value([self.x[-1], self.y[-1]], self.tl[-1])
                    hit_edges = True
            elif E < self.Emin:
                if hit_edges:
                    self._log(
                        "zero field stop encountered at t={:.3e}, x={:.3e}, y={:.3e}, E={:3e}".format(
                            self.t, x, y, E
                        )
                    )
                    break
                else:
                    self._log(
                        "zero field encountered at t={:.3e}, x={:.3e}, y={:.3e}, E={:3e}, going backwards".format(
                            self.t, x, y, E
                        )
                    )
                    self.backwards()
                    r.set_initial_value([self.x[-1], self.y[-1]], self.tl[-1])
                    hit_edges = True
            # streamline loop stop
            elif np.sqrt(
                (x - self.x[0]) ** 2 + (y - self.y[0]) ** 2
            ) < 0.1 * self.R and abs(self.t) > (20 * abs(self.dt)):
                self._log("field loop stop at t={:.3e}".format(self.t))
                break
        self.sort()

    def s(self):
        return np.cumsum(np.sqrt(np.diff(self.x) ** 2 + np.diff(self.y) ** 2))

    def spos(self, coef):
        if self.len > 1:
            return np.searchsorted(self.s(), self.s()[-1] * coef)
        else:
            return 0

    def pos(self, i):
        return self.x[i], self.y[i]

    def meanpos(self, i, p):
        if i + p < self.len:
            n1 = i
            n2 = i + p
        elif i - p >= 0:
            n1 = i - p
            n2 = i
        elif i + 1 < self.len:
            n1 = i
            n2 = i + 1
        elif i - 1 > 0:
            n1 = i - 1
            n2 = i
        else:
            n1 = 0
            n2 = 1
        return (np.mean(self.x[n1:n2]), np.mean(self.y[n1:n2]))

    def add_arrows(self, positions=[], **kwargs):
        """add arrows along the streamline at positions in arrow_positions
        if positions not specified, add to default positions"""
        if not positions:
            positions = self.arrow_positions
        for c in positions:
            n = self.spos(c)
            arrow_tail = self.pos(n)
            arrow_head = self.meanpos(n, 2)
            if n > 0:
                p = mpatches.FancyArrowPatch(arrow_tail, arrow_head, **kwargs)
                self.patches.append(p)


class Figure:
    """Création de la figure à partir de la donnée de
    Figure(charges, xmin, xmax, ymin, ymax, enlarge=3000, verbose = True)

    charges est une liste d'objets Charge()
    xmin... sont les bornes de la figure
    enlarge pour l'intégration des lignes de champ (cf Streamline)

    méthodes utiles :
        set_SL_automatic_starting_points(R=0.2, N=16, dt=0.01)
            crée les SL à calculer automatiquement

        append_other_SL(streamlines = [])
            en rajoute si besoin

        remove_SL(indexes = []):
            ou en enlève

        compute_streamlines()

        compute_arrows(arrowstyle='fancy',mutation_scale=0.3, **kwargs)
            met les flèches à tous les endroits désirés
            avec les kwargs de FancyArrowPatch
    """

    def __init__(self, charges, xmin, xmax, ymin, ymax, verbose=True):
        self.charges = charges  # list of charges objetcs
        self.streamlines = []  # will contain the list of the streamline objects
        self.figure_limits = [xmin, xmax, ymin, ymax]
        self.verbose = verbose
        self.setdefault = False

        # Potentiel calculations :
        self.xgrid = None
        self.ygrid = None
        self.Vtot = None
        self.Vmin = 0
        self.Vmax = 0
        self.levels = None
        self.contours = None

    def _log(self, s, **kwargs):
        if self.verbose:
            print(s, **kwargs)

    def __len__(self):
        return len(self.streamlines)

    def set_default_SL_integration_parameters(
        self, R=0.2, dt=0.01, enlarge=1.5, arrow_positions=[0.5], Emax=1000, Emin=0.0001
    ):
        self.default_parameters = dict(
            dt=dt,
            R=R,
            enlarge=enlarge,
            arrow_positions=arrow_positions,
            Emax=Emax,
            Emin=Emin,
            verbose=self.verbose,
        )

    def create_SL_with_automatic_starting_points(self, R=0.2, N=16):
        """Start around each charges at distances R, evenly spaced in angle coordinates.
        if dipole:not evenly spaced
        R and N can be tuned to refine plotting
        """

        def NLdistrib(x):
            return (
                np.arccos(1 - 2 * np.mod(x, np.pi) / np.pi)
                + np.floor(x / np.pi) * np.pi
            )

        identifier = self.__len__()
        for C in self.charges:
            if isinstance(C, (Charge, Ligne)):
                for alpha in np.linspace(0, 2 * np.pi, num=N, endpoint=False):
                    sl = Streamline(
                        C.x + np.cos(alpha) * R,
                        C.y + np.sin(alpha) * R,
                        name="{:02d}".format(identifier),  #
                        **self.default_parameters
                    )
                    self.streamlines.append(sl)
                    identifier += 1
            elif isinstance(C, Dipole):
                for alpha in np.linspace(0, 2 * np.pi, num=N, endpoint=True):
                    sl = Streamline(
                        C.x + np.cos(-C.alpha + NLdistrib(alpha)) * R,
                        C.y + np.sin(-C.alpha + NLdistrib(alpha)) * R,
                        name="{:02d}".format(identifier),
                        **self.default_parameters
                    )
                    self.streamlines.append(sl)
                    identifier += 1

    def append_new_SL(self, x0, y0, **kwargs):
        params = {}
        for key, val in self.default_parameters.items():
            if key in kwargs:
                params[key] = kwargs[key]
            else:
                params[key] = val
        name = "{:02d}".format(self.__len__())
        sl = Streamline(x0, y0, name=name, **params)
        self.streamlines.append(sl)
        self._log("New SL{:s} ".format(name) + params.__str__())

    def tweak_SL(self, identifier, **kwargs):
        self.streamlines[identifier].change_parameter(**kwargs)

    def remove_SL(self, indexes=[]):
        """remove the indexed streamlines in list indexes
        useful to remove automatic computed SL
        identify with number of SL"""

        n = self.__len__()
        for j in indexes:
            assert 0 <= j < n and type(j) is int, "index not in streamlines"
        indexes.sort()
        for i, j in enumerate(indexes):
            self._log("SL{:s} bypassed".format(self.streamlines[j - i].name))
            self.streamlines.pop(j - i)

    def _get_patches(self):
        """concatenate all patches"""
        p = []
        for sl in self.streamlines:
            for patch in sl.patches:
                p.append(patch)
        return p

    def _get_patches_with_name(self):
        """concatenate all patches"""
        p = []
        i = 0
        for sl in self.streamlines:
            for patch in sl.patches:
                p.append((i, patch, sl.name))
                i += 1
        return p

    def total_field(self, x, y):
        Ex, Ey = 0, 0
        for C in self.charges:
            E = C.E(x, y)
            Ex = Ex + E[0]
            Ey = Ey + E[1]
        return [Ex, Ey]

    def total_norm(self, x, y, param=None):
        """ param necessary to maintain compatibility """
        Ex, Ey = self.total_field(x, y)
        return np.sqrt(Ex ** 2 + Ey ** 2)

    def total_field_direction(self, t, y):
        Ex, Ey = self.total_field(y[0], y[1])
        n = max(np.sqrt(Ex ** 2 + Ey ** 2), 0.0001)
        return [Ex / n, Ey / n]

    def total_potential(self, x, y):
        V = 0
        for C in self.charges:
            V += C.V(x, y)
        return V

    def compute_streamlines(self):
        """All STREAMLINES INTEGRATION"""
        self._log("Integrate field lines...")
        for sl in self.streamlines:
            sl.integrate_streamlines(
                self.total_field_direction, self.total_norm, self.figure_limits
            )

    def compute_arrows(self, arrowstyle="fancy", arrowsize=0.3, **kwargs):
        self._log("Compute arrows positions...")
        for sl in self.streamlines:
            sl.add_arrows(arrowstyle=arrowstyle, mutation_scale=arrowsize, **kwargs)

    def plot_streamlines(self, ax, **kwargs):
        """PLOT streamlines"""
        for sl in self.streamlines:
            ax.plot(sl.x, sl.y, '-k', **kwargs)

    def plot_arrows(self, ax, **kwargs):
        """ Plot arrows"""
        if 'color' not in kwargs:
            kwargs['color'] = 'k'
        collection = PatchCollection(self._get_patches(), zorder=10, **kwargs)
        ax.add_collection(collection)

    def display_streamline_identifier(
        self, ax, R=0.2, format_string=None, fontsize=8, **kwargs
    ):
        """add SL index (name) text next to arrows
        R is distance to anchor"""
        if format_string is None:
            format_string = "{:s}"
        slpositions = self._get_patches_with_name()
        plen = len(slpositions)
        for i, p, name in slpositions:
            Abl, Aur = (
                p.get_path().get_extents().get_points()
            )  # bounding box of the arrow in data coordinates
            x, y = (
                0.5 * Aur
                + 0.5 * Aur
                + [
                    0.5 * R * np.cos(2 * np.pi * i / plen),
                    0.5 * R * np.sin(2 * np.pi * i / plen),
                ]
            )
            ax.text(x, y, format_string.format(name), fontsize=fontsize, **kwargs)

    def compute_potential(self, gridsize=512):
        # POTENTIAL GRID
        self._log("Compute potentials...", flush=True)
        xmin, xmax, ymin, ymax = self.figure_limits
        x = np.linspace(xmin, xmax, gridsize)
        y = np.linspace(ymin, ymax, gridsize)
        self.xgrid, self.ygrid = np.meshgrid(x, y)
        self.Vtot = self.total_potential(self.xgrid, self.ygrid)
        self.Vmin = np.amin(self.Vtot)
        self.Vmax = np.amax(self.Vtot)

    def set_automatic_levels(
        self,
        Nlevels=20,
        logscale=False,
        symmetric=False,
        Vmin=None,
        Vmax=None,
        deltaV=1e-3,
    ):
        self._log("Compute levels automatically...", flush=True)
        self._log("Options:", end="")
        if Vmin is None:
            Vmin = self.Vmin
        if Vmax is None:
            Vmax = self.Vmax
        if logscale:
            self._log("Logscale ", end="")
            if symmetric:
                self._log("symmetric")
                levels = np.power(
                    10,
                    np.linspace(
                        np.log10(deltaV),
                        np.log10(max([abs(Vmin), abs(Vmax)])),
                        num=Nlevels,
                    ),
                )
                self.levels = np.append(-levels[::-1], levels)
            else:
                self._log("not symmetric")
                self.levels = np.logspace(np.log10(Vmin), np.log10(Vmax), num=Nlevels)
        else:
            self._log("Linscale ", end="")
            if symmetric:
                self._log("symmetric")
                levels = np.linspace(deltaV, max([abs(Vmin), abs(Vmax)]), num=Nlevels)
                self.levels = np.append(-levels[::-1], levels)
            else:
                self._log("not symmetric")
                self.levels = np.linspace(Vmin, Vmax, num=Nlevels)
        self._log("V Levels: ", end="")
        for i in self.levels:
            self._log("{:.2e}, ".format(i), end="")
        return self.levels

    def set_levels(self, levels=[]):
        self.levels = levels

    def add_levels(self, levels=[]):
        self.levels = list(self.levels)+levels
        self.levels.sort()

    def plot_contours(
        self, ax, contraste=3, cmap=mpl.cm.RdYlBu_r, scale=None, **kwargs
    ):
        contourscale = 10 ** contraste
        if not scale:
            scale = mpl.colors.Normalize(
                vmin=self.Vmin / contourscale, vmax=self.Vmax / contourscale
            )
        self.contours = ax.contour(
            self.xgrid,
            self.ygrid,
            self.Vtot,
            levels=self.levels,
            cmap=cmap,
            norm=scale,
            **kwargs
        )
        return scale

    def plot_contours_label(self, ax, inline=1, fontsize=10):
        ax.clabel(self.contours, inline=inline, fontsize=fontsize)

    def plot_charges(self, ax, chargesize=12, dipolesize=12):
        # plot point charges
        xmin, xmax, ymin, ymax = self.figure_limits
        dipolesize = dipolesize / 100  # size of the arrow
        for C in self.charges:
            radius = (
                chargesize / (72 * 10) * (xmax - xmin)
            )  # chargesize in points roughly scaled to data values
            if C.q > 0:
                color = "r"
            else:
                color = "b"
            circle = plt.Circle(
                (C.x, C.y), radius=radius * np.sqrt(np.abs(C.q)), color=color, ec="gray"
            )
            artist = ax.add_artist(circle)
            artist.set_zorder(10)
            if isinstance(C, Dipole):
                dx = radius * np.sin(C.alpha)
                dy = radius * np.cos(C.alpha)
                ax.arrow(
                    C.x - dx / 2,
                    C.y - dy / 2,
                    dx,
                    dy,
                    head_width=dipolesize,
                    head_length=dipolesize,
                    fc="k",
                    ec="k",
                    zorder=10,
                    length_includes_head=True,
                )
