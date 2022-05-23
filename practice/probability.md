# Probability & Statistics 

<span style="font-size:2em;">Questions</span>

```{contents}
:local:
```

<!-- Add a margin note using the following code snippet -->
<!-- 
````{margin}
```{note}
Here's my note!
```
````
-->

## Warm-Ups 

### What is the mean of a random variable? How is it computed? 

The expected value, $\mathbb{E}(X)$ is the mean, for a discrete variable,

$$\mathbb{E}(X) = \sum_x x\mathbb{P}(X=x)$$ 

and for continuous variables 

$$ \mathbb{E}(X) = \int_{-\infty}^\infty xp(x)dx$$

### What is the variance of a random variable? 

The variance is a measure of spread of a random variable, defined as 

$$\mathbb{E}[(X-\mu_X)^2]$$ 

where $\mu_X$ is the mean of $X$. 

### When and how do expected value and variance add? The is, what is $\mathbb{E}(aX+bY)$ and $\text{Var}(aX+bY)$?

Expectation is always linear, that is $\mathbb{E}(aX+bY) = a\mathbb{E}(X)+b\mathbb{Y}$. 

Variance only adds when variables are independent, and scalar come out with squared since 

$$\text{Var}(aX) = \mathbb{E}[(aX-a\mu_X)^2] = \mathbb{E}[a^2(X-\mu_X)^2] =  a^2\text{Var}(X)$$

so **if $X$ and $Y$ are independent,

$$\text{Var}(aX+bY) = a^2\text{Var}(X)+b^2\text{Var}(Y).$$

A counterexample for the aditivity of variance when the variables are dependent is $Y=-X$, then $\text{Var}(X+Y) = \text{Var}(0)=0\neq 2\text{Var}(X)$ for any non-constant $X$. 

### What is Bayes Rule? 

Bayes Rules: For two events $A$ and $B$, 

$$ P(A|B) = \frac{P(B|A)P(A)}{P(B)} $$

### What does it mean for two events to independent? 

$A,B$ are independent $\iff$ $P(A\text{ and }B) = P(A)P(B)$

### What is the Law of Total Probability? 

Given a collection of disjoint events, $B_1,\ldots, B_n$ with $B_1\sqcup \cdots \sqcup B_n$ covering the entire sample space,

$$ P(A) = P(A|B_1)+\cdots + P(A|B_n) $$

### What is a probability mass function (pmf)? probability density function (pdf)? Cumulative distribution function (cdf)?

For a discrete random variable $X$, the probability mass function is 

$$ p(a) = \mathbb{P}(X=a)$$

For a continuous random variable, $\mathbb{P}(X=a)=0$$, so we define the continuous analog, the probability density function such that

$$ \mathbb{P}(a\leq X\leq b) = \int_a^b p(x)dx$$

The cumulative distribution function, defined for both continuous and discrete R.V.s, is 

$$ F(a) = \mathbb{P}(X\leq a)$$

### What are the relationships between pdfs and cdfs? What properties do they have? 

Given a pdf $p$ and a cdf $F$, 

$$ F(a) = \mathbb{P}(X\leq a) = \int_{-\infty}^a p(x)dx $$

Some properties of pdfs: 
* $p(x) \geq 0$ for all $x$ 
* $p(x)$ may be greater than $1$ (unlike a pmf)
* $\int_{-\infty}^\infty p(x)dx=1$ 

Some properties of cdfs:
* $F(x)\in [0,1]$ with $\lim_{x\rightarrow \infty}F(x) = 1$ and $\lim_{x\rightarrow -\infty}F(x) = 0$ 
* $F$ is monotonically increasing 
* $F$ is continuous from the right (but not necessarily the left)

### What is a Bernoulli Trial? What is its mean and variance? 

A Bernoulli trial is a single experiment with a "success" outcome occuring with probability $p$ (ex: coin flip).

The mean is given by 

$$ \mathbb{E}(X) = 0(1-p) + 1(p) =p $$ 

The variance is 

$$ \mathbb{E}((X-p)^2) = (0-p)^2(1-p) + (1-p)^2(p) = p(1-p)$$

### What is the binomial distribution? What are its parameters? 
 Given $n$ Bernoulli trials, let $X$ be the number of successes that occured, then $X$ has a binomial distribution, $X\sim Binom(n,p)$. 

This is a **discrete** distribution, with probability mass function
$$ P(X=k) = {n \choose k} p^{k}(1-p)^{n-k}. $$

where ${n \choose k}=\frac{n!}{k!(n-k)!}$ is the binomial coeffient.

The expected value is the sum of the expected value for each Bernoulli trial, which is $p$, so $\mathbb{E}(X)=np$. Similarly, given the indepedence of the trials, the variance is $\text{Var}(X) = np(1-p)$. 

### What is the Poisson distribution? What are its parameters?

### What is the Uniform distribution? What are its parameters?

### What is the Exponential distribution? What are its parameters?

### What is the Normal distribution? What are its parameters?

### How is variance defined? 

### What is covariance? What about the correlation $\rho$ between $X$ and $Y$?

### What is the Law of Large Numbers? 

### State the Central Limit Theorem. 

### What are Type I and Type II errors? How do they relate to the significance level $\alpha$ and the power level $\beta$?


## Probability 

### If you roll 3 dice in order, what is the probability that the values will be strictly decreasing? 


### A disease screening test has a 95% accuracy for detecting the disease, and a 2% false positive rate. The odds that a person has the disease are 1%. If a person tests positive, what are the odds they have the disease?  


### What is the probability that a seven-game series goes to 7 games?

### Say you draw a circle and choose two chords at random. What is the probability that those chords will intersect?

### There are 50 cards of 5 different colors. Each color has cards numbered between 1 to 10. You pick 2 cards at random. What is the probability that they are not of same color and also not of same number?

### What is the expected number of rolls needed to see all 6 sides of a fair die?

### Three friends in Seattle each told you it’s rainy, and each person has a 1/3 probability of lying. What is the probability that Seattle is rainy? Assume the probability of rain on any given day in Seattle is 0.25.

### Three ants are sitting at the corners of an equilateral triangle. Each ant randomly picks a direction and starts moving along the edge of the triangle. What is the probability that none of the ants collide? Now, what if it is k ants on all k corners of an equilateral polygon?

### How many cards would you expect to draw from a standard deck before seeing the first ace?

### Say you are given an unfair coin, with an unknown bias towards heads or tails. How can you generate fair odds using this coin?

### A fair die is rolled n times. What is the probability that the largest number rolled is r, for each r in 1,...,6?

### There are two groups of n users, A and B, and each user in A is friends with those in B and vice versa. Each user in A will randomly choose a user in B as their best friend and each user in B will randomly choose a user in A as their best friend. If two people have chosen each other, they are mutual best friends. What is the probability that there will be no mutual best friendships?






## Statistics 

### Derive the expected value for the uniform distribution $[a,b]$? 

### What is the Central Limit Theorem? Why is it useful? 

### How would you explain a confidence interval to a non-technical audience?

### Describe p-values in layman’s terms.

### Describe A/B testing. What are some common pitfalls?

### How would you derive a confidence interval from a series of coin tosses?

### You sample from a uniform distribution [0, d] n times. What is your best estimate of d?

###  You are drawing from a normally distributed random variable X ~ N(0, 1) once a day. What is the approximate expected number of days until you get a value of more than 2?

### Derive the expectation for a geometric distributed random variable.

### Given a random Bernoulli trial generator, how do you return a value sampled from a normal distribution. 

### What is MLE? What about MAP? How do they compare? 

### Explain hypothesis testing. 

### What is a $z$-test? What is a $t$-test? When would you use each? 

### Suppose you draw $n$ samples from a uniform distribution, $U(a,b)$. What are the MLE estimates of $a$ and $b$? 

### Let $X,Y\sim U(0,1)$ be independent, what is the expected value of $\max(X,Y)$?