# sparse_GP
Toy problem to learn sparse GPs

The problem set up is as follows:

$$**y**=f(**x**)+\boldsymbol{\epsilon}$$ 

where **y** and **x** are observed and the function values are to learned and evaluated at new values of **x**. In conventional Gaussian processes, the posterior over functions $p(**f**|**y**)$ is required. This can be calculated by considering the joint Gaussian distribution $p(**y**,**f**)$ and then conditioning on $**y**$ using standard identitites:

$$p(**f**|**y**)=\mathcal{N}\left(K_{TN}K_{NN}^{-1} **y**,K_{TT}-K_{TN}K_{NN}^{-1}K_{NT}\right)$$ 

where there are N training points and f is evaluated at T test points. THe inversion of the matrix $K_{NN}$ limits this method to N<10,000, hence it is not applicable to larger datasets. Therefore sparse GPs must be used.

The idea is to represent the posterior over functions, $p(f|y)$ using a variational distribution $q(f)$:

$$p(**f**|**y**)=\int{p(**f**|**u**)p(**u**|**y**)}d**u**=\int{p(**f**|**u**)q(**u**)}d**u**$$

hence it is actually a variational distribution over the inducing points **u**. This assumes that the distribution over the functions can be represented by essentially fitting a GP to a set of artificial points. 

$$q(**u**)=\mathcal{N}\left(**m**,**S**\right)$$
This distribution is defined in terms of the variational parameters **m**, **S** and **Z**, where **Z** represents the position of the inducing points. To fit the parameters, $q(**u**)$ is optimised to be close to the true posterior over **u** by minimising the KL divergence between the two distributions:

$$KL(q(**u**)||p(**u**|**y**))$$

Expanding this out leads to the ELBO to maximise:

$$ELBO=E_{q(**u**)}[p(**y**|**u**)]-KL(q(**u**)||p(**u**))$$

Given everything is Gaussian, these things can all be evaluated analytically to give an expression to be maximised. This expression forces q to be closer to the prior over **u** but also maximise the likelihood given the observed data.
