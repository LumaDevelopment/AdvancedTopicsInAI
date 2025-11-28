# Report

The first work that is now generally recognized as artificial intelligence (AI) was done by Warren McCulloch and Walter Pitts in 1943 [[1]](#bibliography). Since then, the field has expanded dramatically in breadth, complexity, popularity, and a host of other metrics. In the last decade, this growth has been driven primarily by advances in machine learning (ML), resulting from countless innovations in optimization, computational hardware, neural network architecture, data processing, and other fields. This report surveys essential techniques in optimization and exciting frontiers in AI, enabled in part by those techniques.

ML, while a subfield of AI, is itself a field encompassing a variety of techniques, including supervised learning, unsupervised learning, and reinforcement learning [[1]](#bibliography). In supervised learning, a practitioner designs a model capable of learning a desired function, then presents it with many examples of the function’s input and output, iteratively adjusting the model to minimize the difference between the predicted and real output for any given input. Therefore, supervised learning is simply the traversal of a vast, multidimensional generalized loss space, where with adjustments to model parameters we attempt to find the global minima. The process of this traversal is, fundamentally, the process of optimization.

For many optimization methods, we need the ability to arbitrarily evaluate a function at a given point. However, this is not always possible or practical. In such a case, we may turn to interpolation, where given a set of known points, we draw a smooth curve through them, allowing arbitrary evaluation within the region defined by these points [[2]](#bibliography). We discuss three interpolation methods: linear, polynomial, and cubic spline. We assume interpolation in one dimension, although interpolation can be accomplished in multiple dimensions.

In linear interpolation, for each pair of points, we simply draw a straight line between them. Of course, this method struggles to produce smooth curves and therefore is restricted in the set of functions it can accurately model.

For functions where linear interpolation performs poorly, one may consider polynomial interpolation, where a polynomial function of a given degree is fitted to the given points. While this method works significantly better for smooth functions, the degree of the polynomial significantly affects the accuracy of the model and therefore must be chosen prudently. Additionally, polynomial interpolation suffers from Runge’s phenomenon, where oscillation occurs at the edge of the interpolated interval when interpolating between equispaced points with a high degree polynomial [[3]](#bibliography).

Cubic spline interpolation combines the simplicity of linear interpolation with the flexibility of polynomial interpolation. Instead of fitting a single polynomial to all points, cubic spline interpolation fits a cubic polynomial between each consecutive pair of points. This method inherits the smooth behavior of polynomial interpolation while avoiding Runge’s phenomenon, making it generally preferable over polynomial interpolation.

With the capability for arbitrary evaluation, we now return to optimization, starting with optimization in one dimension. We begin with golden section search, which is a technique for finding an extremum of a function inside a specified interval. The method relies on using the golden ratio to iteratively compute new interior points within the interval, moving the bounds of the interval to the interior points when they indicate proximity to the desired type of extremum (minima or maxima). With this iterative interval adjustment, golden section search progressively closes in on an extremum within the initial interval. While this method is simple, it converges slowly and does little to adapt based on the function’s shape.

Those looking for a more sophisticated approach may turn to Brent’s method. Brent’s method is also an iterative algorithm, but at each step it checks to see if the region of the function it’s examining is sufficiently smooth/parabolic. If so, it uses regular or inverse parabolic interpolation to fit a parabola to that region of the function and immediately jumps to the desired extremum. Otherwise, it falls back to a traditional golden section search. This allows Brent to converge much quicker than golden section search for well-behaved functions, while retaining robustness in other cases.

While Brent is a great choice for derivative-free optimization, if the derivative is available, it can be integrated into Brent’s method for even greater effect. However, there are multiple schools of thought regarding how exactly to use the derivative. One suggests that instead of remaining bounded to parabolic interpolations, Brent can use high order polynomial interpolations that agree with function and derivative evaluations. Another suggests simply using the derivate to educate whether to shrink the extremum search interval from the left or right side. Either way, the use of the derivative can greatly quicken convergence.

Of course, not all problems exist in one dimension, and in the real world we quickly find ourselves in need of optimization methods that operate in multiple dimensions, many which employ the one-dimensional optimization methods we just covered (henceforth referred to as line searches).

One great example of this is Powell’s method. Given an initial point and a set of search directions, the method optimizes the function along each direction in turn, finding the desired extremum along each direction. Then, the method computes a new position using the sum of the previous position and a linear combination of the optimized search directions. The search direction which contributed the most to the new position is dropped from the list of directions, the new linear combination is added to set, and the next iteration of the algorithm begins. This repeats until no significant improvement is made, and the method returns the last computed position.

Like Brent’s method, Powell’s method is a great pick for derivative-free optimization. However, when gradient information is available, more efficient methods can be employed, like conjugate gradient (CG) methods. CG methods iteratively compute search directions that are approximately conjugate to the Hessian of the function and optimize along them one-by-one. Due to this property, CG methods find the extremum of a quadratic form after *N* minimizations, where Powell’s methods perform the same number of optimizations in a single iteration! Additionally, CG methods converge quadratically for all other functions. This makes it a great choice when gradient information is available.

CG methods are also notable because they use storage on the order of *N*. However, if storage on the order of *N x N* is available, quasi-Newton methods may be an even better pick. Quasi-Newton methods find the extremum of a quadratic form after *N* minimizations and converge quadratically for all other functions, just like CG methods. However, they also (in general) converge faster and work on more functions compared to CG methods. This is because while CG methods build directions that are conjugate to the Hessian, quasi-Newton methods iteratively build an approximation of the whole Hessian and use it to optimize with Newton’s method. Because quasi-Newton methods build this approximation, they can make sure that it is always positive definite, meaning it will always guide optimization correctly, which isn’t always true with the real Hessian.

The optimization methods we have discussed until this point are concerned with pure minimization and maximization, where any input is valid. We now turn to linear programming, where we can introduce more precise constraints on our optimization.

Our first method of interest in the simplex method, which solves optimization problems defined by linear equality and inequality constraints and a linear objective function. It operates by traversing the vertices of the feasible space defined by these constraints, moving from one vertex to another in a way that improves the objective at each step. Eventually, it arrives at the optimal vertex if one exists. Although the algorithm has an exponential time complexity, in practice, it often performs much better.

In contrast, interior point methods approach the same class of problems from within the feasible region, rather than along its edges. These methods iteratively follow a path through the interior of the feasible space that leads toward the optimal point, often leveraging Newton-like updates. Interior point methods have polynomial time complexity in theory, and are therefore very effective for large, spare, structured problems.

As our optimization methods grow in complexity, so too must the techniques and tools used in their implementation. Multidimensional optimization and linear programming both involve solving linear systems, and in this requirement, we find an application for matrix decomposition. LU decomposition factors a square matrix A into the product of two matrices: L, a lower triangular matrix, and U, an upper triangular matrix, such that A = LU. Once this decomposition is complete, when solving Ax = b, you can instead solve Ly = b for y and Ux = y for x, splitting the problem into two much easier computations, making it more efficient to solve linear systems.

However, matrix decomposition’s utility is not limited to solving linear systems. One essential data transformation in ML, and particularly in unsupervised learning, is dimensionality reduction. Singular Value Decomposition (SVD) is a method for decomposing a matrix into a product of three matrices derived from the matrix’s singular values and singular vectors (similar to eigenvalues and eigenvectors). In this decomposed form, a given number of features of the matrix can be maintained, and the rest dropped, to create a reduced dimensionality version of the original matrix.

Finally, we note that matrix decomposition can be used for even more interpretable feature extraction. Non-negative matrix factorization (NMF) is a method for decomposing a matrix into a product of two matrices, with one of them capturing the essential features of the data, and the other capturing the modifications to the base features to get the specific matrix. NMF is more constrained than SVD because all the elements of the matrix must be nonnegative. However, because each feature exclusively adds to the data, the features are typically easier to interpret. For instance, if one uses NMF to factorize an image of a face, the features may be the eyes, nose, mouth, etc.

Having established a strong background of mathematical and computational techniques, we now look to the recent history of ML by surveying paradigms, innovations, and models that either pushed forward or demonstrated the power of this burgeoning field.

In April 2006, one-shot learning was introduced by Fei-Fei et al., showing the ability to categorize objects with only one instance of a given class being demonstrated [[4]](#bibliography). Lampert et al. extended this even further in June 2009 with zero-shot learning, demonstrating the ability to classify objects never seen before by identifying features of an image and defining classes as combinations of features [[5]](#bibliography).

In February 2016, Google researchers introduced the federated learning paradigm [[6]](#bibliography). In this implementation of ML, a server distributes an initial model to a set of clients. These clients individually train the model with local data, then send their models back to the server for aggregation. In this paradigm, data is never sent away from the client, allowing for ML to be applied to problems where it previously couldn’t be due to privacy concerns.

In June 2017, Google researchers released the seminal paper “Attention Is All You Need”, introducing the transformer deep learning architecture [[7]](#bibliography). The transformer’s power spawned entire new families of ML models, including large language models, and contributed significantly to the AI boom observed over the last decade.

In December 2018, Google researchers announced it had created AlphaZero, a revolutionary game-playing AI that can teach itself to play games without being given rules or example games [[8]](#bibliography). The model quickly become proficient enough to beat gold standard engines in games such as chess.

In July 2021, Google researchers released AlphaFold 2, a model which performs predictions of protein structure using deep learning [[9]](#bibliography). AlphaFold 2’s approach integrates physical and biological knowledge directly into the ML algorithm, contributing to the model’s astounding accuracy.

In May 2022, Google researchers released the first version of Imagen, a text-to-image diffusion model [[10]](#bibliography). While many such models existed at the time of publication, the model marked the introduction of another major player into the image generation model space and further demonstrated the power of the transformer architecture Google introduced 5 years prior.

In November 2023, Google researchers introduced GraphCast, an AI approach to global weather forecasting [[11]](#bibliography) that predicts weather conditions more accurately and much faster than the industry gold-standard weather simulation system [[12]](#bibliography). Notably, GraphCast uses graph neural networks, which are neural networks that are specialized to process spatially structured data.

Google is a major cradle of innovation in the AI/ML space. But seemingly every day, research organizations and corporations around the world are making massive breakthroughs in the field, enabling what we could’ve never imagined to be possible, and pulling ideas and figments of our collective imagination into reality. Perhaps one day, these models could be running on the very hardware that dreamed up these innovations in the first place [[13]](#bibliography).
 
## Bibliography

[1] S. Russel and P. Norvig, Artificial intelligence: A Modern approach, 4th ed. Prentice Hall, 2020.

[2] W. Press, S. Teukolsky, W. Vetterling, and B. Flannery, Numerical recipes : the art of scientific computing. Cambridge University Press, 2007.

[3] C. Runge, “Über empirische Funktionen und die Interpolation zwischen äquidistanten Ordinaten,” Zeitschrift für Mathematik und Physik, vol. 46, pp. 224–243, 1901.

[4] Li Fei-Fei, R. Fergus, and P. Perona, “One-shot learning of object categories,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 28, no. 4, pp. 594–611, Apr. 2006, doi: https://doi.org/10.1109/tpami.2006.79.

[5] C. H. Lampert, H. Nickisch, and S. Harmeling, “Learning to detect unseen object classes by between-class attribute transfer,” 2009 IEEE Conference on Computer Vision and Pattern Recognition, Jun. 2009, doi: https://doi.org/10.1109/cvpr.2009.5206594.

[6] M. H. Brendan, E. Moore, D. Ramage, S. Hampson, and Blaise, “Communication-Efficient Learning of Deep Networks from Decentralized Data,” arXiv (Cornell University), Feb. 2016, doi: https://doi.org/10.48550/arxiv.1602.05629.

[7] A. Vaswani et al., “Attention Is All You Need,” arXiv, Jun. 12, 2017. https://arxiv.org/abs/1706.03762

[8] D. Silver et al., “A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play,” Science, vol. 362, no. 6419, pp. 1140–1144, Dec. 2018, doi: https://doi.org/10.1126/science.aar6404.

[9] J. Jumper et al., “Highly Accurate Protein Structure Prediction with Alphafold,” Nature, vol. 596, no. 7873, pp. 583–589, Jul. 2021, doi: https://doi.org/10.1038/s41586-021-03819-2.

[10] C. Saharia et al., “Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding,” arXiv:2205.11487 [[cs]](#bibliography), May 2022, Available: https://arxiv.org/abs/2205.11487

[11] R. Lam et al., “Learning skillful medium-range global weather forecasting,” Science, vol. 382, no. 6677, Nov. 2023, doi: https://doi.org/10.1126/science.adi2336.

[12] R. Lam, “GraphCast: AI model for faster and more accurate global weather forecasting,” Google DeepMind, Nov. 14, 2023. https://deepmind.google/discover/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/

[13] B. J. Kagan et al., “In vitro neurons learn and exhibit sentience when embodied in a simulated game-world,” Neuron, vol. 110, no. 23, Oct. 2022, doi: https://doi.org/10.1016/j.neuron.2022.09.001.
