# Basic Mathmetics

ID | Name | Commits
---|---|--- 
00 | [Linear Algebra Gilbert Strang MIT 18.065]() | --

---

terminology|expression
---|---
discrete | $\sum$
continuous | $\int$
--- | $log\prod \Rightarrow \sum$


---
## Linear Algebra


---
## Calculus
- Integration

- Differentiation
    - x
 
- [matrixcalculus online](http://www.matrixcalculus.org/)

---
## Probability

- Event $x$ or $A$
- Random Variable $X$
- Probability $p(x)$
- Probability distribution $P(X)$
- density function $f_X(x)$
- 
Terminology | Symbol | Commits
---|---|---
sample space | $\Omega$ | all possible outcomes
event space | $\Sigma$ | all possible events; depends on how to define  an experiment on sample space 
random variable | $X$ | A Random Variable is a function that maps outcomes to real values, but reality is that the application of Random Variables is more connected to the events; discrete and continuous
probability distribution | $P(X=x_i)=p_i$ | e.g Bernoulli distribution
indpendent events| -- | --
joint events | -- | --
conditional events | $P(A/B)$ | the likelihood of event $A$ occurrring given that $B$ is true;
distribution mass function | -- | -- 
Bayes's theory | $P(A/B)=\frac{P(B/A)P(A)}{P(B)}$ | A, B are events; Consider a sample space $Ω$ generated by two random variables $X$ and $Y$, the events $A = \{X = x\}$ and $B = \{Y = y\}$.
normal distribution | $X\sim{N(\mu,\sigma^2)}$
mean/expectation  | -- | -- 
variance | -- | --

reference:
- https://towardsdatascience.com/but-what-is-a-random-variable-4265d84cb7e5
- https://medium.com/@aerinykim

---
## Statistics
- Expectation $E(X)$
- Variance $Var(X)$

Concept | Expression | Commits
---|---|---
expectation | -- | --
variance | -- | --


---
## Imformation Thoery

| Concept | Expression |  Commits |
| --- | --- | --- | 
Information | $I(x)=-logp_x$| $x$ is Event; Low probabilty == Higt information, same otherwise;
Entropy | $H(X)=-\sum_{x\subseteq X}p(x)logp(x)$ | Uncentainty
-- | $H(X,Y)=H(X)+H(Y)-I(X,Y)=-E[log p(x,y)]=-\sum_x\sum_yp(x,y)logp(x,y)$
Conditional Entropy | $H(Y/X)=H(X,Y)-H(X)=\sum_xp(x)H(Y/x)=-\sum_xp(x)\sum_yp(y/x)log p(y/x)=\sum_{x,y}p(x,y)logp(y/x)$ |
Relative Entropy/KL Divergence| $D_{KL}(p\|q)=H_p(q)-H(p)$ | $D_{KL}(p\|q)\neq D_{KL}(q\|p)$ 
Cross Entropy| $CEH(p,q)=H_p(q)=E_p(-logq)=H(p)+D_{KL}(p\|q)$ | --
Mutual Information | $I\left ( X,Y \right )=H(Y)-H(Y/X)= \sum _{x\subseteq X}\sum_{y\subseteq Y}p\left ( x,y \right )log\frac{p(x,y)}{p(x)p(y)}$ | $I(X,Y)\geq 0, I(X,Y)=I(Y,X)$ 

- Entropy
    - 
    - $H(X)\geqslant0$
    - $H(X,Y)=H(X)+H(Y), if(X,Y),independent$
    - $H(X,Y)=H(X)+H(Y)-I(X,Y),elsewise$

---

