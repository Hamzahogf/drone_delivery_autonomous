from __future__ import annotations
from typing import List, Optional
import numpy as np

# raisimpy is the python binding for raisimlib.
try:
    import raisimpy as raisim
except ImportError:
    raisim = None

class VisualMarkerManager:
    """
        Manages visiual markers and HUD text for the delivery environment.

        In GUI mode the manager registers coloured visual spheres (one per
        client) and three debug-text items with the RaisimServer that streams
        to raisimUnity. In headless mode every method is a no-op.

        server:
            A live raisim.RaisimServer instance (or None for headles).
        gui_mode:
            Set to False to skip all rendering calls.
    """

    _COLOUR_PENDING = (1.0, 0.0, 0.0, 1.0) # red - not yet delivered
    _COLOUR_DELIVERED = (0.0, 1.0, 0.0, 1.0) # green - delivered

    # HUD label identifiers (used as unique names in raisimUnity)
    _HUD_NAMES = ["hud_energy", "hud_remaining", "hud_target"]

    def __init__(
            self,
            server: Optional[raisim.RaisimServer],
            gui_mode: bool = True,
    ) -> None:
        self.server = server
        self.gui_mode = gui_mode and server is not None
        # per-client state
        self._sphere_names: List[str] = []
        self._label_names: List[str] = []
        self._delivery_status: List[bool] = []
        self._client_positions: List[np.ndarray] = []
        # visual shpere radius (meters) - scaled from the delivery radius
        self._vis_radius: float = 0.15
        self._hud_initialized: bool = False
        self._hud_spheres: List[object] = []
        # obstacle sphere state (rebuilt every episode via sync_obstacles)
        self._obstacle_names: List[str] = []
        self._OBSTACLE_COLOUR = (0.55, 0.55, 0.55, 0.9)  # grey
    
    @staticmethod
    def _vis_radius_from_delivery_radius(delivery_radius: float) -> float: 
        """Map physics delivery radius -> on-screen sphere radius (clamped)"""
        return float(np.clip(delivery_radius * 0.12, 0.08, 0.6))
    
    def _unique_sphere_name(self, prefix: str, index: int) -> str:
        """Generate a unique name for a visual sphere based on client index."""
        return f"{prefix}_sphere_{index}"
    
    def _add_sphere(
            self,
            name: str,
            position: np.ndarray,
            colour: tuple,
            radius: float
    ) -> object:
        """Create (or recreate) a named visual sphere on the RaisimServer."""
        r, g, b, a = colour
        sphere = self.server.addVisualSphere(name, radius, r, g, b, a)
        sphere.setPosition(position.astype(float))
        return sphere
    
    def _remove_sphere(self, name: str) -> None:
        """Remove a named visual object from the server (best-effort)"""
        try:
            self.server.removeVisualObject(name)
        except Exception as e:
            pass
    
    def set_delivery_radius(
            self,
            delivery_radius: float,
            client_positions: np.ndarray,
            delivered_mask: np.ndarray,
            num_clients: int
    ) -> None:
        """
            recreate client markers sized to the current adaptive delivery radius.

            Called by the environment whenever the curriculum anneals the
            catchment radius so the visualisation statys consistent.
        """
        if not self.gui_mode:
            return
        self._vis_radius = self._vis_radius_from_delivery_radius(delivery_radius)
        self.sync_markers_to_state(
            client_positions,
            delivered_mask,
            num_clients,
            self._vis_radius
        )
    
    def sync_markers_to_state(
            self,
            client_positions: np.ndarray,
            delivered_mask: np.ndarray,
            num_clients: int,
            visual_sphere_radius: float
    )-> None:
        """
        Rebuild all client markers from scratch.

        Removes every existing marker and recreates them so colour,
        position, and size are all consistent with the current state.
        """
        if not self.gui_mode:
            return

        self._vis_radius = visual_sphere_radius

        # Remove old spheres and labels
        for name in self._sphere_names + self._label_names:
            self._remove_sphere(name)
        self._sphere_names = []
        self._label_names = []
        self._delivery_status = []
        self._client_positions = []

        for i in range(num_clients):
            pos = client_positions[i].astype(float)
            delivered = bool(delivered_mask[i])
            colour = self._COLOUR_DELIVERED if delivered else self._COLOUR_PENDING

            sphere_name = self._unique_sphere_name("client_sphere", i)
            self._add_sphere(sphere_name, pos, colour, visual_sphere_radius)
            self._sphere_names.append(sphere_name)

            # Small white label sphere placed slightly above the delivery marker
            label_name = self._unique_sphere_name("client_label", i)
            label_pos  = pos.copy()
            label_pos[2] += 0.35
            self._add_sphere(label_name, label_pos, (1.0, 1.0, 1.0, 0.6), 0.05)
            self._label_names.append(label_name)
 
            self._delivery_status.append(delivered)
            self._client_positions.append(pos.copy())

    def sync_obstacles(
            self,
            obstacle_positions: np.ndarray,
            obstacle_radii: np.ndarray,
    ) -> None:
        """
        Rebuild obstacle spheres from scratch for the current episode.
        Called once per reset() — obstacle count/positions/radii change
        every episode under domain randomization.
        """
        if not self.gui_mode:
            return
        for name in self._obstacle_names:
            self._remove_sphere(name)
        self._obstacle_names = []

        for i in range(len(obstacle_positions)):
            name = self._unique_sphere_name("obstacle", i)
            self._add_sphere(
                name,
                obstacle_positions[i].astype(float),
                self._OBSTACLE_COLOUR,
                float(obstacle_radii[i]),
            )
            self._obstacle_names.append(name)

    def reset(
            self, client_positions: np.ndarray, delivery_radius: float = 0.5
    ) -> None:
        """ 
        inintialize visual markers for all client(all pending/red).
        args:
            client_positions: 3D positions of all clients (shape: [num_clients, 3])
            delivery_radius: radius for visual spheres (scaled from physics radius)
        """
        if not self.gui_mode:
            return
        n = len(client_positions)
        mask = np.zeros(n, dtype=bool)  # all pending
        vr = self._vis_radius_from_delivery_radius(delivery_radius)
        self.sync_markers_to_state(client_positions, mask, n, vr)
        self._setup_hud()

    def _setup_hud(self) -> None:
        """
        Create (or recreate) three tiny HUD marker spheres.

        raisimUnity shows the object name in the scene overlay, whihch we use
        to diplay live telemetry; The spheres are placed high in the scene
        (z= 8m) so they don't interfere with the arena.
        """
        if not self.gui_mode:
            return
        # Only remove if already initialized (they exist on server)
        if self._hud_initialized:
            for name in self._HUD_NAMES:
                self.server.removeVisualObject(name)

        self._hud_spheres = []
        hud_positions = [
            np.array([-4.0, 4.0, 8.0]),
            np.array([-4.0, 3.5, 8.0]),
            np.array([-4.0, 3.0, 8.0]),
        ]
        for name, pos in zip(self._HUD_NAMES, hud_positions):
            s = self._add_sphere(name, pos, (1.0, 1.0, 0.0, 0.8), 0.05)
            self._hud_spheres.append(s)
        self._hud_initialized = True

    def mark_delivered(self, client_index: int) -> None:
        """
        Change a client's marker from red (pending) to green (delivered).
        Args:
        client_index: index of the client to update
        """
        if not self.gui_mode or client_index >= len(self._sphere_names):
            return
        if self._delivery_status[client_index]:
            return  # already green
        self._delivery_status[client_index] = True
        name = self._sphere_names[client_index]
        pos = self._client_positions[client_index]

        # replace with a green sphere
        self._remove_sphere(name)
        self._add_sphere(name, pos, self._COLOUR_DELIVERED, self._vis_radius)

    def update_hud(self, energy, remaining, total, target_index=None):
        if not self.gui_mode:
            return
        target_str = f"C{target_index}" if target_index is not None else "None"
        new_names = [
            f"Energy:{energy:.1f}%",
            f"Rem:{remaining}/{total}",
            f"Tgt:{target_str}",
        ]
        hud_positions = [
            np.array([-4.0, 4.0, 8.0]),
            np.array([-4.0, 3.5, 8.0]),
            np.array([-4.0, 3.0, 8.0]),
        ]
        new_spheres = []
        for old, new_name, pos in zip(self._HUD_NAMES, new_names, hud_positions):
            if self._hud_initialized:          # ← only remove if they exist
                self.server.removeVisualObject(old)
            s = self._add_sphere(new_name, pos, (1.0, 1.0, 0.0, 0.8), 0.05)
            new_spheres.append(s)
        self._HUD_NAMES = new_names
        self._hud_spheres = new_spheres
        self._hud_initialized = True
    
    def cleanup(self) -> None:
        """Remove all visual markers and HUD objects from the server."""
        if not self.gui_mode:
            return
        for name in self._sphere_names + self._label_names + self._obstacle_names:
            self._remove_sphere(name)
        for name in self._HUD_NAMES:
            self._remove_sphere(name)
        self._sphere_names    = []
        self._label_names     = []
        self._delivery_status = []
        self._client_positions = []
        self._hud_spheres     = []
        self._obstacle_names  = []