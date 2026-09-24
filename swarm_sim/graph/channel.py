from typing import Tuple, Dict, Any, Optional
import numpy as np
import networkx as nx
from swarm_sim.utils.enums import AgentStatus

class ChannelModel:
    """Shakhatreh et al. (2021) Air-to-Ground (A2G) & Cellular Backhaul Path Loss & Throughput Model."""
    def __init__(
        self,
        freq_hz: float = 2.4e9,
        tx_power_dbm: float = 20.0,
        bandwidth_hz: float = 10e6,
        noise_floor_dbm: float = -104.0,
        a_los: float = 9.61,
        b_los: float = 0.16,
        eta_los_db: float = 1.0,
        eta_nlos_db: float = 20.0,
        uav_altitude_m: float = 30.0
    ):
        self.freq_hz = freq_hz
        self.tx_power_dbm = tx_power_dbm
        self.bandwidth_hz = bandwidth_hz
        self.noise_floor_dbm = noise_floor_dbm
        self.a_los = a_los
        self.b_los = b_los
        self.eta_los_db = eta_los_db
        self.eta_nlos_db = eta_nlos_db
        self.uav_altitude_m = uav_altitude_m
        
        # Speed of light
        self.c = 3e8
        
        # Free space path loss constant at 1m: 20*log10(4*pi*f/c)
        self.fspl_1m_db = 20.0 * np.log10(4.0 * np.pi * self.freq_hz / self.c)

    def compute_a2g_path_loss(self, dist_2d: float, h: Optional[float] = None) -> float:
        """Compute elevation-dependent A2G / Relay-Relay path loss in dB."""
        h_m = h if h is not None else self.uav_altitude_m
        d_3d = np.sqrt(dist_2d**2 + h_m**2)
        d_3d = max(d_3d, 1.0)  # Avoid log(0)
        
        # Elevation angle in degrees
        theta_deg = np.degrees(np.arctan2(h_m, max(dist_2d, 1e-3)))
        
        # LoS probability
        p_los = 1.0 / (1.0 + self.a_los * np.exp(-self.b_los * (theta_deg - self.a_los)))
        
        # Free space component
        fspl_db = 20.0 * np.log10(d_3d) + self.fspl_1m_db
        
        pl_los = fspl_db + self.eta_los_db
        pl_nlos = fspl_db + self.eta_nlos_db
        
        pl_avg = p_los * pl_los + (1.0 - p_los) * pl_nlos
        return float(pl_avg)

    def compute_cellular_backhaul_path_loss(self, dist_2d: float) -> float:
        """Compute cellular base station to UAV backhaul path loss in dB."""
        d_m = max(dist_2d, 1.0)
        pl_cell = 28.0 + 22.0 * np.log10(d_m) + 20.0 * np.log10(self.freq_hz / 1e9)
        return float(pl_cell)

    def compute_achievable_rate(self, path_loss_db: float) -> Tuple[float, float]:
        """Compute SNR (dB) and achievable data rate (Mbps)."""
        rx_power_dbm = self.tx_power_dbm - path_loss_db
        snr_db = rx_power_dbm - self.noise_floor_dbm
        
        # Linear SNR
        snr_lin = 10.0 ** (snr_db / 10.0)
        
        # Shannon capacity: R = B * log2(1 + SNR) in Mbps
        rate_mbps = (self.bandwidth_hz * np.log2(1.0 + snr_lin)) / 1e6
        return float(snr_db), float(rate_mbps)

    def get_link_throughput(
        self,
        pos_i: np.ndarray,
        pos_j: np.ndarray,
        is_backhaul: bool = False
    ) -> Tuple[float, float]:
        """Return (snr_db, rate_mbps) for a link between positions pos_i and pos_j."""
        dist_2d = float(np.linalg.norm(pos_i - pos_j))
        if is_backhaul:
            pl_db = self.compute_cellular_backhaul_path_loss(dist_2d)
        else:
            pl_db = self.compute_a2g_path_loss(dist_2d)
            
        return self.compute_achievable_rate(pl_db)

    def compute_network_throughput(self, sim: Any, graph: nx.Graph) -> float:
        """Compute sum-rate (Mbps) across all active links in the communication graph."""
        total_sum_rate = 0.0
        pos = sim.positions
        
        for u, v in graph.edges():
            is_backhaul = (sim.roles[u].value.startswith("ground") or sim.roles[v].value.startswith("ground"))
            _, rate_mbps = self.get_link_throughput(pos[u], pos[v], is_backhaul=is_backhaul)
            total_sum_rate += rate_mbps
            
        return float(total_sum_rate)

    def precompute_r_ref(self, num_relays: int, dist_ab: float = 223.6) -> float:
        """Precompute reference sum-rate (Mbps) for optimal uniform spacing topology."""
        spacing = dist_ab / (num_relays + 1)
        
        # A chain of (num_relays + 2) nodes separated by spacing
        total_rate = 0.0
        for _ in range(num_relays + 1):
            pl_db = self.compute_a2g_path_loss(spacing)
            _, rate = self.compute_achievable_rate(pl_db)
            total_rate += rate
            
        return float(total_rate)
