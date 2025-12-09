"""Blade operations inspired by pygae/clifford."""

import functools
import itertools
import operator

import torch


# copied from the itertools docs
def _powerset(iterable):
    """Generate powerset of an iterable.
    
    Example: powerset([1,2,3]) -> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3).
    """
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


class ShortLexBasisBladeOrder:
    """Short lexicographic basis blade ordering for Clifford algebras."""
    
    def __init__(self, n_vectors):
        """Initialize blade ordering for n basis vectors.
        
        Args:
            n_vectors: Number of basis vectors in the algebra.
        """
        self.index_to_bitmap = torch.empty(2**n_vectors, dtype=int)
        self.grades = torch.empty(2**n_vectors, dtype=int)
        self.bitmap_to_index = torch.empty(2**n_vectors, dtype=int)

        for i, t in enumerate(_powerset([1 << i for i in range(n_vectors)])):
            bitmap = functools.reduce(operator.or_, t, 0)
            self.index_to_bitmap[i] = bitmap
            self.grades[i] = len(t)
            self.bitmap_to_index[bitmap] = i
            del t  # enables an optimization inside itertools.combinations


def set_bit_indices(x: int):
    """Iterate over the indices of bits set to 1 in x, in ascending order."""
    n = 0
    while x > 0:
        if x & 1:
            yield n
        x = x >> 1
        n = n + 1


def count_set_bits(bitmap: int) -> int:
    """Count the number of bits set to 1 in bitmap."""
    count = 0
    for i in set_bit_indices(bitmap):
        count += 1
    return count


def canonical_reordering_sign_euclidean(bitmap_a, bitmap_b):
    """Compute the sign for the product of bitmap_a and bitmap_b.
    
    Assumes a Euclidean metric.
    
    Args:
        bitmap_a: First bitmap.
        bitmap_b: Second bitmap.
    
    Returns:
        Sign (+1 or -1) for the product.
    """
    a = bitmap_a >> 1
    sum_value = 0
    while a != 0:
        sum_value = sum_value + count_set_bits(a & bitmap_b)
        a = a >> 1
    if (sum_value & 1) == 0:
        return 1
    else:
        return -1


def canonical_reordering_sign(bitmap_a, bitmap_b, metric):
    """Compute the sign for the product of bitmap_a and bitmap_b.
    
    Uses the supplied metric for computation.
    
    Args:
        bitmap_a: First bitmap.
        bitmap_b: Second bitmap.
        metric: Metric array.
    
    Returns:
        Sign for the product.
    """
    bitmap = bitmap_a & bitmap_b
    output_sign = canonical_reordering_sign_euclidean(bitmap_a, bitmap_b)
    i = 0
    while bitmap != 0:
        if (bitmap & 1) != 0:
            output_sign *= metric[i]
        i = i + 1
        bitmap = bitmap >> 1
    return output_sign


def gmt_element(bitmap_a, bitmap_b, sig_array):
    """Compute element of the geometric multiplication table for blades a and b.
    
    Implementation described in ga4cs chapter 19.
    
    Args:
        bitmap_a: Bitmap for blade a.
        bitmap_b: Bitmap for blade b.
        sig_array: Signature array.
    
    Returns:
        Tuple of (output_bitmap, output_sign).
    """
    output_sign = canonical_reordering_sign(bitmap_a, bitmap_b, sig_array)
    output_bitmap = bitmap_a ^ bitmap_b
    return output_bitmap, output_sign


# def construct_gmt(index_to_bitmap, bitmap_to_index, signature):
#     n = len(index_to_bitmap)
#     array_length = int(n * n)
#     coords = torch.zeros((3, array_length), dtype=torch.uint8)
#     k_list = coords[0, :]
#     l_list = coords[1, :]
#     m_list = coords[2, :]

#     # use as small a type as possible to minimize type promotion
#     mult_table_vals = torch.zeros(array_length)
#     inner_table_vals = torch.zeros(array_length)
#     outer_table_vals = torch.zeros(array_length)

#     for i in range(n):
#         bitmap_i = index_to_bitmap[i]

#         for j in range(n):
#             bitmap_j = index_to_bitmap[j]
#             bitmap_v, mul = gmt_element(bitmap_i, bitmap_j, signature)
#             v = bitmap_to_index[bitmap_v]

#             list_ind = i * n + j
#             k_list[list_ind] = i
#             l_list[list_ind] = v
#             m_list[list_ind] = j

#             mult_table_vals[list_ind] = mul
#             if v + 1 == abs(i - j):
#                 inner_table_vals[list_ind] = mul
#             if v + 1 == i + j + 2:
#                 outer_table_vals[list_ind] = mul
                

#     return torch.sparse_coo_tensor(indices=coords, values=mult_table_vals, size=(n, n, n)), \
#             torch.sparse_coo_tensor(indices=coords, values=inner_table_vals, size=(n, n, n)), \
#             torch.sparse_coo_tensor(indices=coords, values=outer_table_vals, size=(n, n, n))

import numpy as np
def construct_gmt(index_to_bitmap, bitmap_to_index, signature):
    """Construct geometric multiplication tables.
    
    Args:
        index_to_bitmap: Index to bitmap mapping.
        bitmap_to_index: Bitmap to index mapping.
        signature: Metric signature.
    
    Returns:
        List of [t_geom, t_inner, t_outer] tensors.
    """
    n = len(index_to_bitmap)
    array_length = int(n * n)

    t_geom = np.zeros((n, n, n), dtype=np.int32)
    t_inner = np.zeros((n, n, n), dtype=np.int32)
    t_outer = np.zeros((n, n, n), dtype=np.int32)
    
    
    for i in range(n):
        bitmap_i = index_to_bitmap[i]

        for j in range(n):
            bitmap_j = index_to_bitmap[j]
            bitmap_v, mul = gmt_element(bitmap_i, bitmap_j, signature)
            v = bitmap_to_index[bitmap_v]

            t_geom[i,j,v] = mul
            if v+1 == abs(i - j):
                t_inner[i,j,v] = mul
            if v+1 == i + j +2:
                t_outer[i,j,v] = mul
                
    return  [torch.tensor(_) for _ in [t_geom,t_inner,t_outer]]
        