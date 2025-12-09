"""Clifford algebra implementation based on DavidRuhe/clifford-group-equivariant-neural-networks."""

import functools
import math

import torch
from torch import nn

from torch_ga.clifford.blades import ShortLexBasisBladeOrder, construct_gmt, gmt_element
from torch_ga import GeometricAlgebra, MultiVector

class CliffordAlgebra(nn.Module):
    """Clifford algebra implementation with PyTorch support.
    
    Provides geometric product, involutions, and grade operations.
    """
    
    def __init__(self, metric):
        """Initialize the algebra from a metric tensor.
        
        Args:
            metric: Diagonal metric values defining the algebra.
        """
        super().__init__()

        self.register_buffer("metric", torch.as_tensor(metric.detach() if isinstance(metric, torch.Tensor) else metric))
        self.num_bases = len(metric)
        self.bbo = ShortLexBasisBladeOrder(self.num_bases)
        self.dim = len(self.metric)
        self.n_blades = len(self.bbo.grades)
        cayley, cayley_inner, cayley_outer  = [_ for _ in construct_gmt(self.bbo.index_to_bitmap, self.bbo.bitmap_to_index, self.metric)]
        cayley, cayley_inner, cayley_outer  = [ _.to_dense().to(torch.get_default_dtype()) for _ in [cayley, cayley_inner, cayley_outer]] 
        self.grades = self.bbo.grades.unique()
        self.register_buffer(
            "subspaces",
            torch.tensor(tuple(math.comb(self.dim, g) for g in self.grades)),
        )
        self.n_subspaces = len(self.grades)
        self.grade_to_slice = self._grade_to_slice(self.subspaces)
        self.grade_to_index = [
            torch.tensor(range(*s.indices(s.stop))) for s in self.grade_to_slice
        ]

        self.register_buffer(
            "bbo_grades", self.bbo.grades.to(torch.get_default_dtype())
        )
        self.register_buffer("even_grades", self.bbo_grades % 2 == 0)
        self.register_buffer("odd_grades", ~self.even_grades)
        self.register_buffer("cayley", cayley)
        self.register_buffer("cayley_inner", cayley_inner)
        self.register_buffer("cayley_outer", cayley_outer)
        
    def geometric_product(self, a, b, blades=None):
        """Compute the geometric product of two multivectors."""
        return self.product(a, b, blades, _type="geometric")
    
    def inner_product(self, a, b, blades=None):
        """Compute the inner product of two multivectors."""
        return self.product(a, b, blades, _type="inner")
    
    def outer_product(self, a, b, blades=None):
        """Compute the outer product of two multivectors."""
        return self.product(a, b, blades, _type="outer")
    
    def product(self, a, b, blades=None, _type="geometric"):
        """Compute specified product type between multivectors."""
        if _type in ["geometric"]:
            cayley = self.cayley
        elif _type in ["inner"]:
            cayley = self.cayley
        elif _type in ["outer"]:
            cayley = self.cayley_outer
        else:
            raise Exception(f"Unknownd product {_type}")

        if blades is not None:
            blades_l, blades_o, blades_r = blades
            assert isinstance(blades_l, torch.Tensor)
            assert isinstance(blades_o, torch.Tensor)
            assert isinstance(blades_r, torch.Tensor)
            cayley = cayley[blades_l[:, None, None], blades_o[:, None], blades_r]

        return torch.einsum("...i,ijk,...k->...j", a, cayley, b)

    def __call__(self, a: torch.Tensor) -> MultiVector:
        """Create a MultiVector from a geometric algebra tensor.

        Args:
            a: Geometric algebra tensor to return MultiVector for.

        Returns:
            MultiVector for the input tensor.
        """
        if False: a = a.to(dtype=torch.float32)
        # return MultiVector(a, GeometricAlgebra(self.metric.detach().numpy()))
        return MultiVector(a, GeometricAlgebra(self.metric))
    
    def to_ga(self) -> GeometricAlgebra:
        """Create a GeometricAlgebra from the Clifford Algebra.

        Returns:
            GeometricAlgebra instance.
        """
        return GeometricAlgebra(self.metric)
    
    def _grade_to_slice(self, subspaces):
        """Convert grade to slice for indexing into multivector."""
        grade_to_slice = list()
        subspaces = torch.as_tensor(subspaces)
        for grade in self.grades:
            index_start = subspaces[:grade].sum()
            index_end = index_start + math.comb(self.dim, grade)
            grade_to_slice.append(slice(index_start, index_end))
        return grade_to_slice

    @functools.cached_property
    def _alpha_signs(self):
        """Sign factors for main involution."""
        return torch.pow(-1, self.bbo_grades)

    @functools.cached_property
    def _beta_signs(self):
        """Sign factors for reversion."""
        return torch.pow(-1, self.bbo_grades * (self.bbo_grades - 1) // 2)

    @functools.cached_property
    def _gamma_signs(self):
        """Sign factors for Clifford conjugation."""
        return torch.pow(-1, self.bbo_grades * (self.bbo_grades + 1) // 2)

    def alpha(self, mv, blades=None):
        """Clifford main involution."""
        signs = self._alpha_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv.clone()

    def beta(self, mv, blades=None):
        """Clifford main anti-involution (reversion)."""
        signs = self._beta_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv.clone()

    def gamma(self, mv, blades=None):
        """Clifford conjugation."""
        signs = self._gamma_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv.clone()

    def zeta(self, mv):
        """Extract scalar (grade-0) component."""
        return mv[..., :1]

    def embed(self, tensor: torch.Tensor, tensor_index: torch.Tensor) -> torch.Tensor:
        """Embed tensor into multivector at specified blade indices."""
        mv = torch.zeros(
            *tensor.shape[:-1], 2**self.dim, device=tensor.device, dtype=tensor.dtype
        )
        mv[..., tensor_index] = tensor
        mv = mv.to(dtype=torch.float32)
        return mv

    def embed_grade(self, tensor: torch.Tensor, grade: int) -> torch.Tensor:
        """Embed tensor into multivector at specified grade."""
        mv = torch.zeros(*tensor.shape[:-1], 2**self.dim, device=tensor.device)
        s = self.grade_to_slice[grade]
        mv[..., s] = tensor
        return mv

    def get(self, mv: torch.Tensor, blade_index: tuple[int]) -> torch.Tensor:
        """Extract specific blade coefficients from multivector."""
        blade_index = tuple(blade_index)
        return mv[..., blade_index]

    def get_grade(self, mv: torch.Tensor, grade: int) -> torch.Tensor:
        """Extract grade-k component from multivector."""
        s = self.grade_to_slice[grade]
        return mv[..., s]

    def b(self, x, y, blades=None):
        """Compute the symmetric bilinear form (inner product).

        Args:
            x: First multivector.
            y: Second multivector.
            blades: Blade indices. Defaults to None.

        Returns:
            Scalar part of symmetric product.
        """
        if blades is not None:
            assert len(blades) == 2
            beta_blades = blades[0]
            blades = (
                blades[0],
                torch.tensor([0]),
                blades[1],
            )
        else:
            blades = torch.tensor(range(self.n_blades))
            blades = (
                blades,
                torch.tensor([0]),
                blades,
            )
            beta_blades = None

        return self.geometric_product(
            self.beta(x, blades=beta_blades),
            y,
            blades=blades,
        )

    def q(self, mv, blades=None):
        """Compute the scalar square of a multivector.

        Args:
            mv: Input multivector.
            blades: Blade indices. Defaults to None.

        Returns:
            Scalar square of the multivector.
        """
        if blades is not None:
            blades = (blades, blades)
        return self.b(mv, mv, blades=blades)

    def _smooth_abs_sqrt(self, input, eps=1e-16):
        """Compute smooth abs then sqrt for differentiable norm."""
        return (input**2 + eps) ** 0.25

    def norm(self, mv, blades=None):
        """Compute the Clifford norm of a multivector."""
        return self._smooth_abs_sqrt(self.q(mv, blades=blades))

    def norms(self, mv, grades=None):
        """Compute norms for each grade component."""
        if grades is None:
            grades = self.grades
        return [
            self.norm(self.get_grade(mv, grade), blades=self.grade_to_index[grade])
            for grade in grades
        ]

    def qs(self, mv, grades=None):
        """Compute scalar squares for each grade component."""
        if grades is None:
            grades = self.grades
        return [
            self.q(self.get_grade(mv, grade), blades=self.grade_to_index[grade])
            for grade in grades
        ]

    def reverse(self, mv, blades=None):
        """Perform the reversion operation on multivectors, an operation specific to geometric algebra.

        In Geometric Algebra, the reverse of a multivector is formed by reversing the order of the vectors in each blade.

        Args:
            mv (torch.Tensor): Input multivectors.
            blades (Union[tuple, list, torch.Tensor], optional): Specify which blades are present in the multivector.

        Returns:
            torch.Tensor: The reversed multivector.
        """
        grades = self.bbo.grades.to(mv.device)
        if blades is not None:
            grades = grades[torch.as_tensor(blades, dtype=int)]
        signs = torch.pow(-1, torch.floor(grades * (grades - 1) / 2))
        return signs * mv.clone()
    
    def sandwich(self, u, v, w=None):
        """Compute sandwich product. If w is None returns uv~u, else uvw."""
        if w is None:
            return self.sandwich2(u,v)
        else:
            return self.sandwich3(u,v,w)
                
    def sandwich2(self, a, b):
        """Compute sandwich product aba'."""
        return self.geometric_product(self.geometric_product(a, b), self.reverse(a))
    
    def sandwich3(self, u, v, w):
        """Compute triple product uvw."""
        return self.geometric_product(self.geometric_product(u, v), w)

    def output_blades(self, blades_left, blades_right):
        """Compute output blade indices from product of left and right blades."""
        blades = []
        for blade_left in blades_left:
            for blade_right in blades_right:
                bitmap_left = self.bbo.index_to_bitmap[blade_left]
                bitmap_right = self.bbo.index_to_bitmap[blade_right]
                bitmap_out, _ = gmt_element(bitmap_left, bitmap_right, self.metric)
                index_out = self.bbo.bitmap_to_index[bitmap_out]
                blades.append(index_out)

        return torch.tensor(blades)

    def random(self, n=None):
        """Generate n random multivectors."""
        if n is None:
            n = 1
        return torch.randn(n, self.n_blades)

    def random_vector(self, n=None):
        """Generate n random grade-1 vectors."""
        if n is None:
            n = 1
        vector_indices = self.bbo_grades == 1
        v = torch.zeros(n, self.n_blades, device=self.cayley.device)
        v[:, vector_indices] = torch.randn(
            n, vector_indices.sum(), device=self.cayley.device
        )
        return v

    def parity(self, mv):
        """Determine if multivector is odd (True) or even (False) grade."""
        is_odd = torch.all(mv[..., self.even_grades] == 0)
        is_even = torch.all(mv[..., self.odd_grades] == 0)

        if is_odd ^ is_even:  # exclusive or (xor)
            return is_odd
        else:
            raise ValueError("This is not a homogeneous element.")

    def eta(self, w):
        """Return parity sign factor for a homogeneous multivector."""
        return (-1) ** self.parity(w)

    def alpha_w(self, w, mv):
        """Apply parity-dependent transformation based on versor w."""
        return self.even_grades * mv + self.eta(w) * self.odd_grades * mv

    def inverse(self, mv, blades=None):
        """Compute the multiplicative inverse of a multivector."""
        mv_ = self.beta(mv, blades=blades)
        return mv_ / self.q(mv)

    def rho(self, w, mv):
        """Apply the versor w action to mv (reflects in hyperplane normal to w)."""
        return self.sandwich3(w, self.alpha_w(w, mv), self.inverse(w))

    def reduce_geometric_product(self, inputs):
        """Reduce a sequence of multivectors via geometric product."""
        return functools.reduce(self.geometric_product, inputs)

    def versor(self, order=None, normalized=True):
        """Generate a random versor from product of random vectors."""
        if order is None:
            order = self.dim if self.dim % 2 == 0 else self.dim - 1
        vectors = self.random_vector(order)
        versor = self.reduce_geometric_product(vectors[:, None])
        if normalized:
            versor = versor / self.norm(versor)[..., :1]
        return versor

    def rotor(self):
        """Generate a random even-grade versor element."""
        return self.versor()

    @functools.cached_property
    def geometric_product_paths(self):
        """Compute which grade combinations produce non-zero output."""
        gp_paths = torch.zeros((self.dim + 1, self.dim + 1, self.dim + 1), dtype=bool)

        for i in range(self.dim + 1):
            for j in range(self.dim + 1):
                for k in range(self.dim + 1):
                    s_i = self.grade_to_slice[i]
                    s_j = self.grade_to_slice[j]
                    s_k = self.grade_to_slice[k]

                    m = self.cayley[s_i, s_j, s_k]
                    gp_paths[i, j, k] = (m != 0).any()

        return gp_paths

