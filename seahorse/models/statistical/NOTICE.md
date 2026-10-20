# Statistical source attribution

The Hawkes source and parameter conventions in this directory were copied
from the independently maintained `TheHawkesPackage`, release `v1.0.0`, commit
`37e83bc30b3bd5781551ea1fc1e72ba657608ebb`, by Jens Fischer:
<https://github.com/jeMATHfischer/TheHawkesPackage>.

The frozen source snapshot is in the private hawkes source directory. In
particular, the port preserves its conventions: the exponential temporal
kernel is `alpha * exp(-beta * s)` (branching ratio `alpha / beta`), the
Omori--Utsu kernel is `alpha * (s + c)**(-p)`, and the Gaussian/Pareto spatial
kernels integrate to one over the plane. The vendored source files remain
unchanged. Seahorse's surrounding adapter adds the independent-sequence runner
contract, analytic Gaussian rectangle mass and compensators, an H-event
conditional composite fitting/scoring path, and an exact rectangle-truncated
Gaussian mixture location sampler. Source Ogata time thinning is retained.
The Omori/Pareto route retains source quadrature and finite MCMC locations.
These adaptations are explicit; the port is not presented as the original
authors' implementation of any Hawkes paper. The source manifest and native
adapter are both included in the frozen campaign source snapshot.

The upstream project is MIT licensed.  Its MIT license applies to material
copied or adapted from it; Seahorse retains its project license for original
code in this directory.

Copyright (c) 2019-2026 Jens Fischer.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
