import gudhi.clustering.tomato as gdt  # For tomato clustering algorithm
import robust_laplacian as rlap
from scipy.sparse.linalg import eigsh


class SegmentMesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        self.neighborhood_graph = None

    def compute_laplacian(self, k: int = None):
        self.laplacian, self.mass = rlap.mesh_laplacian(
            self.vertices, self.faces
        )
        if k is not None:
            self.eigenvalues, self.eigenvectors = eigsh(
                self.laplacian, k, self.mass, sigma=1e-7
            )

    def build_neighbourhood_graph(self):
        """ """
        num_vertices = len(self.vertices)
        self.neighborhood_graph = [[] for _ in range(num_vertices)]

        for i1, i2, i3 in self.faces:
            self.neighborhood_graph[i1].extend([i2, i3])
            self.neighborhood_graph[i2].extend([i1, i3])
            self.neighborhood_graph[i3].extend([i1, i2])

    def __call__(self, weights=None):
        if self.neighborhood_graph is None:
            raise RuntimeError(
                "Neighbourhood graph not defined. "
                "Call `build_neighbourhood_graph()`."
            )
        tomato = gdt.Tomato(graph_type="manual", density_type="manual")
        tomato = tomato.fit(X=self.neighborhood_graph, weights=weights)

        return tomato
