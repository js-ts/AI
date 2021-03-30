


# https://ajcr.net/Basic-guide-to-einsum/

"""
1. Repeating letters between input arrays means that values along those axes will be multiplied together. The products make up the values for the output array.
2. Omitting a letter from the output means that values along that axis will be summed.
3. We can return the unsummed axes in any order we like.
"""


# https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/

"""
1. Free indices are the indices used in the output specification. They are associated with the outer for-loops.
2. Summation indices are all other indices: those that appear in the argument specifications but not in the output specification. They are so called because they are summed out when computing the output tensor. They are associated with the inner for-loops.

the role of each is clear: Free indices are those used to iterate over every output element, while summation indices are those used to compute and sum over the product terms required.

"""

# https://rockt.github.io/2018/04/30/einsum
"""

"""
