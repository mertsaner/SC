### A Pluto.jl notebook ###
# v0.19.13

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 7a7c2ec0-c57f-11ec-12f5-c776dc208f4d
using PlutoVista, PlutoUI, Plots, PyPlot, Colors, LinearAlgebra, ExtendableSparse, SparseArrays, ExtendableGrids, GridVisualize, VoronoiFVM, Triangulate, SimplexGridFactory, DataFrames, Printf, Interact

# ╔═╡ d4467ceb-4bdb-4290-a87a-7aa83df9b8d5
PlutoUI.TableOfContents()

# ╔═╡ b917dad1-c4ef-44ea-a8a8-f40fadc5da1b


# ╔═╡ 1b99ce43-d892-47a3-ad1d-7452aa7f29b7
md"
># Gas Transport in Porous Medium
>#### Scientific Computing, Winter Semester 2021/22
>#### By Prof. Jürgen Fuhrmann
>###### Group members: Garam Kim, Umut Polat, Rosmi Francis
"

# ╔═╡ 8ac82a3f-8a18-4a94-b92c-ea9bd35c892f
md"## 1. Introduction "

# ╔═╡ 9674b832-b4a5-48a7-a759-cd65090dc7d5


# ╔═╡ 50881738-5731-43e1-b9d1-8b368c1b9ce7
md" Gas Transport in a porous medium is a parabolic equation and the problem is transient. Therefore, the problem is time-dependent in the specified domain. In this report, firstly the problem is introduced, and then the possible solutions are given by applying the finite volume method '**VoronoiFVM**'. Neumann Boundary condition is used with two initial values at the boundaries. Additionally, we introduce a solution method to decrease the error. Finally, we present the graphs of solution and error.

"

# ╔═╡ 427e0684-f4b2-491a-aed0-e5488b25279d
md"### 1.1. Physical Background and the Porous Medium Equation"

# ╔═╡ ba49d380-d648-4ac7-bdbe-bd5a3fdb11a4
md"When we consider a gas transport in a porous medium, we can observe advective and diffusive transport. The advective transport is described by Darcy's Law which states: 

$u_g = - \frac{k}{\mu} \nabla p \;\;\;\;(\mathrm{I})$

where k is the permeability of the medium, μ is the viscosity and p is the pressure.

We can simplify the equation of Darcy's law as below since $k$ and $\mu$ are negligible:
Where the gas velocity $u_g$ is proportional to the Gas-phase pressure gradient $∇P_{g}$.

$\vec{u_g}= -∇P_g\;\;\;\; (\mathrm{II})$ 
"

# ╔═╡ fa0db37f-dddf-49db-9d9d-7108461a2620
md" There are different kinds of methods to describe gas diffusion in the porous medium. In our project we will use Stefan-Maxwell Equations to explain free molecule diffusion in a porous medium by      ($J=-D∇u$)."

# ╔═╡ 09ef294d-3885-44db-8bbf-af065b86c40c
md"Gas flowing through a porous medium can be described with the equation below for $m>1$:

$∂_tu + ∇(u^m) = 0\;\;\;\; (\mathrm{II})$  

For $m=2$ the equation ($\mathrm{II}$) has an exact solution that is called the Barenblatt solution. The Barenblatt solution can be described as:

$u(x,t)=max(0,t^{-\alpha}(1-\frac{\alpha(m-1)r^2}{2dmt^{\frac{2\alpha}{d}}}))$
where $r=|x|$ and $α = \frac{1}{m-1+\frac{2}{d}}$. $u(x,t)$ is time depend local amount of species.


The full form of the Porous Medium equation is with a source function on the right-hand side and it is a nonlinear diffusion equation of parabolic type. It can be written as:

$∂_tu = ∇(D(u)∇u)+f \;\;\; (\mathrm{III})$

For the case of $f=0$, $\nabla u$ describes the complete derivate of u in x. The Function $D(u)=mu^{m-1}$ called diffusion coefficient and the condition of nonnegativity of $D$ is needed to make the equation formally parabolic. And We will use Finite Volume Discretization to solve the equation $(\mathrm{III}).$
"

# ╔═╡ 0f014d9e-048a-4baa-a1d3-a6c44cbdedc8
md"## 2. Solution methods for discrete system"

# ╔═╡ 37d357c6-2e0a-4ac5-b987-a916ab188074
md"
Let $A(u)$ be a nonlinear operator $A: D \rightarrow \mathbb{R}^n$, where $D \subset \mathbb{R}^n$ is its domain of definition. Then $A(u) = f(u)$ is called a nonlinear problem. It is well known that if the problem is linear, then there is a powerful solution method, which is Gaussian elimination. For nonlinear problems, however, there is no such a method, so we introduce some methods as below: 



- **Fixpoint iteration scheme**, Assume $A(u)=M(u)u$, where $M(u)$ is linear at each $u$, choose an initial value $u_0$ and at each iteration step, solve: 

$M(u^i)u^{i+1}=f$ 

terminate if the residual is small enough to reach the desired accuracy. This method has a large domain of convergence that may be slow. 


- **Newton iteration scheme**, Let $A'(u)=(a_{kl})$ where $A'(u)$ is the _Jacobi matrix_ of first partial derivatives of $A$ at point $u$, where

$A'(u) = (\frac{\partial}{\partial u_l}A_k(u_1\dots u_n))$.

In the $i$-th iteration step:

$u_{i+1}=u_i - (A'(u_i))^{-1}(A(u_i) -f)$

Or solve the algorithm step by step, one can split the above equation as 
1) Calculate residual: $r_i=A(u_i)-f$, 
2) solve linear system for update: $A'(u_i)h_i = r_i$, 
3) update solution: $u_{i+1}=u_i-h_i$

Compare to the fixed point method, this has a potentially small domain of convergence with a good initial value. Moreover, initial convergence is possibly slow, but when the iteration is close enough to the solution, it has quadratic convergence. Choosing a good initial guess is a big question in the Newton problem, therefore the Modified Newton's method was introduced to speed up its slow convergence and to increase the convergence radius:

- **Damped Newton iteration**, it does not use the full update but damps parameter less than 1 which we increase during the iteration process. Line search also can be used (automatic detection of a damping factor).

" 

# ╔═╡ f9153d35-5189-40f9-8761-f113687eac36
md"## 3. Discretization"

# ╔═╡ e394b22c-299a-406f-b14a-accd0467c3ec
md"### 3.1. Discretization in space"

# ╔═╡ c5401ec5-60e0-4b74-a7d9-bb9f613c720f
md"In order to approximate the equation ($\mathrm{III}$) we need to discretize our domain. By using this method we are dividing the domain into a finite number of closed subsets.

* Let N be the number of discretization points and $h=\frac{L}{N-1}$ where L is the domain length. 
* And $x_i = (i-1)h$  while  $i=1,\cdots,N$ is the discretization points.

_Finite Difference Approximation of the first Derivatives:_

$u'(x_{i+\frac{1}{2}})≈ \frac{u_{i+1}-u_i}{h}$ 


By using boundary values $u_0$ and $u_{N+1}$, we obtain **$n×n$** matrix for 1D. For 2D, we get $n^2×n^2$ matrix with grid points $x_{ij}=h_x(i-1)$  and   $h_y(j-1).$

In order to solve the matrices we need to use iterative methods. Some of these are; the Jacobi method, Gauss-Seidel method, and Incomplete LU decomposition that are implemented in Julia packages such as **IncompleteLU.jl**, **IterativeSolvers.js** etc.

"

# ╔═╡ 141006b1-c91d-410b-946b-20253d90e9a5
md"#### 3.1.1. Creating Discrete System of Equations"

# ╔═╡ 0339e9e2-4f55-4d93-ba24-5db761fff285
md" A standart PDE calculus happens in Lipschitz domains. A domain $Ω ⊂ \mathbb{R}^n$
is a Lipschitz domain $∀x∈∂Ω$, there exists a neighborhood of $x$ on $∂Ω$ which can be represented as the graph of a Lipschitz continuous function, when a function $f : D → \mathbb{R}^m$ is called _Lipschitz continuous_ if there exists $c>0$ such that $||f(x)-f(y)|| ≤ c||x-y||$ for any $x,y\in D$. 

First, we need to write the governing equations at Representative Elementary Volumes (REVs).
We can express the change of the amount of species in $w$ during $t_0$ and $t_1$ by equation ($IV$) below. Where $u(t)$ is the function of the amount of the species, $j(t)$ is the flux function and $f(t)$ is the source function at $w$ respectively. 

$u(t_1)-u(t_0) + \int_{t_0}^{t_1} j(t) . dt = \int_{t_0}^{t_1}f(t).dt\;\;\;\; (\mathrm{IV})$
" 

# ╔═╡ 0c64f543-cc13-4b7c-aaf1-182ff3934d3b
md" 
The flux of the gas species **$j(t)$** going through the boundary of REV **($∂w$)** can be calculated by the equation ($V$) below. 

$\vec{j}=-δ\vec{∇}u\;\;\;\; (\mathrm{V})$    
where δ is either constant, space-dependent, or depends on $u$. For simplicity, we assume δ to be constant. The negative sign is because the flux $\vec{j}$ is proportional to the change of $\vec{u}(\vec{x},t)$ in the negative direction.
"

# ╔═╡ 58377172-d60b-47ed-b71c-c80775aee2e8
md"
Using Gauss Theorem, equation ($\mathrm{V}$) can be written as:

$∂_tu(\vec{x},t)+ ∇.\vec{j}(\vec{x},t)=f(\vec{x},t)\;\;\;\; (\mathrm{VI})$ 


Then putting $\vec{j}$ into the equation ($\mathrm{VI}$):


$∂_tu(\vec{x},t) - ∇.(δ\vec{∇}.\vec{u}(\vec{x},t))=f(\vec{x},t)\;\;\;\; (\mathrm{VII})$

If the above equation ($\mathrm{VII}$) is convective, the flux ($\vec{j}$) will be equal to $-δ\vec{∇}u+u\vec{v}$ where $v$ is the convective velocity.
"

# ╔═╡ 14c7cbde-1dc0-42c6-871e-6ca70b538b77
md" --- "

# ╔═╡ 5107f984-5200-4e8f-a37b-293492901df9
md"#### 3.1.2. Triangulation"

# ╔═╡ 214da77c-e799-49d6-9bbe-3b8d59ed7d94
md"To discretizate our boundary we need to use mesh generation for a 2D domain as triangles and quadrilaterals. In the 2D domain, the boundary of the domain is $∂Ω$ and it includes a finite number of hyperplanes in $R^n$.

To solve the mentioned problem, the Voronoi diagram will be used to create the **admissible grids**. A grid ${T_1...T_M}$ is admissible **if and only if**:
1.  $Ω$ is the union of the elementary cells $(∪_{i=1}^{N_Γ})$. 
2.  $T_m∩ T_n$ consists of exactly one point and this point is a common vertex of $T_m$ and $T_n$.
3.  For $m\neq n$, $T_m ∩T_n$ consists of more than one point and $T_m ∩ T_n$ is a common edge of $T_m$ and $T_n$."

# ╔═╡ 82b09d1e-aff1-470b-a63c-c73a78981488
function dela(;n=10)
	triin=Triangulate.TriangulateIO()
	triin.pointlist = rand(Cdouble,2,n)
	(triout, vorout) = triangulate("Q", triin)
	triin, triout
end

# ╔═╡ 3acc2c69-ea0b-453b-8c03-71269f07043e
triin,triout=dela(n=10);

# ╔═╡ 7eb263d2-76c8-4666-b6ee-b15751a87527
plot_in_out(PyPlot,triin,triout);gcf().set_size_inches(6,4);gcf()

# ╔═╡ 1dde31c5-76f1-4a9f-91d7-70f881d91344
md"The Delaunay triangulation connects the points that have shared edges in the Voronoi diagram in a way that the boundary of the triangle is surrounded by a circle and the edge points (vertices) are situated on the circle's circumference.

Thanks to the Voronoi diagram we can create admissible grids in which the distance between the two points inside the domain is minimized by creating Voronoi point sets that subdivide the whole space into 'nearest neighbor' regions. 

"

# ╔═╡ 5b179258-f50f-4d17-a6da-8ecd0971dba4
function voronoi(;pts=rand(Cdouble,2,8),show_tria=true,circumcircles=false)
	triin=Triangulate.TriangulateIO()
	triin.pointlist = pts
	(triout, vorout) = triangulate("vQ", triin)
	if !show_tria
	   triout.trianglelist=zeros(Cint,2,0)
	end
	plot_in_out(PyPlot,triin,triout,voronoi=vorout,circumcircles=circumcircles)
	gcf().set_size_inches(6,4)
	gcf()
end

# ╔═╡ ce8d5a51-8cc7-40eb-a28e-5fc8b8dbf74a
voronoi(pts=rand(Cdouble,2,10),circumcircles=true,show_tria=true)

# ╔═╡ 237e76fd-da0a-45fd-bf9d-27672111a709
md"By using **Triangulate.jl** package we create Voronoi diagram, the Delaunay triangulation, and the boundary of the point set."

# ╔═╡ efd01d2f-cd90-4b3b-88d4-50f970f9fdaa
md"---- "

# ╔═╡ 68137fb4-0a6b-4ff5-bbbc-6cc7a15ef488
 md"#### 3.1.3. Boundary Conditions"

# ╔═╡ 135757e2-7f08-424c-b448-8913cd7f7e4e
md"There are 3 different Boundary Conditions that can be used. Namely:

* Dirichlet boundary condition: (fixed solution at the boundary)
$u(x)=g_i(x),   \forall\vec{x} ∈ Γ_i$ 
* Neumann boundary condition: (fixed boundary flux)

$δ∇u(\vec{x},t).\vec{n} = g_i(\vec{x},t),\;\;\;\;\;  ∀\vec{x}∈ Γ_i$
* Robin boundary condition: (boundary flux is proportional to the solution)
$δ∇u(\vec{x},t).\vec{n}+ α_i(\vec{x},t)u(\vec{x},t) = g_i(\vec{x},t)$ 

Assuming $∂Ω = ∪_{i=1}^{N_Γ}$ $Γ_i$ is the union of a finite number of non-intersecting subsets $Γ_i$ which are locally Lipschitz.
"

# ╔═╡ 85cf5f67-3127-4141-ad0f-07c2d86bd6ac
md"#### 3.1.4. Constructing the Control Volume "

# ╔═╡ 91be1132-faa6-4597-a875-0efe8a706499
md"Firstly we need to find an algorithm to construct our control volume. Now **VoronoiFVM.jl** package has to be implemented in our domain. Assume that our domain $Ω$ (assuming $Ω$ ⊂ $\mathbb{R}^d$ is a polygonal domain such that $∂Ω=∪_{m∈G} Γ_m$, where $Γ_m$ are planar such that $\vec{n}$|$_{Γ_m} = \vec{n_m}$)  is subdivided into finite number of REVs.

By subdividing $Ω$ into finite number of control volumes, we get $\overline{\Omega}= ∪_{a\in N}\overline{w_a}$. After that we need to assign a value of $u$ at each $w$ which is $u_i$ (this value is calculated at collocation points $x_i∈w$). Then we will find the PDE approximation.

*  $w_a$ are open convex domains such that $w_a ∩ w_l = ∅$ if $w_a \neq w_b$
*  $σ_{ab}=\overline{w_a} ∩ \overline{w_b}$ are either empty, points or straight lines. If $|σ_{ab}|>0$ we say that $w_a$ and $w_b$ are neighbors.
*  $\vec{n_{ab}} ⟂ σ_{ab}$ : normal of $∂w_a$ at $σ_{ab}$
*  $\mathcal{N_a} = {l ∈ \mathcal{N} : |\sigma_{ab}| > 0}$ : set of neighbours of $w_a$
*  $γ_{am} = ∂w_a ∩ Γ_m$ : boundary part of $∂w_a$
*  $\mathcal{G_a}= {m ∈ \mathcal{G}: |γ_{am}|> 0}$:  set of non-empty boundary parts of $∂w_a$
$∂w_a= (\cup_{b\in\mathcal{N_a}}\sigma_{ab})\cup(\cup_{m\in\mathcal{G_a}}\gamma_{am})$
→ Intersection with neighboring control volume : $(\cup_{b\in\mathcal{N_a}}\sigma_{ab})$

→ Possible intersection with boundary of domain : $(\cup_{m\in\mathcal{G_a}}\gamma_{am})$


The picture below from [Juliahub](The Voronoi finite volume method) shows a schematic view of the finite volume approach. 
![](https://drive.google.com/uc?export=view&id=1opI8VNzMqkVa9F3DQsgf6bncQJIlvkGv)
"

# ╔═╡ 775ad737-7022-4545-abb5-5d8c03818088
md" From the diagram above, we can see that the domains $w_a$ and $w_b$ are not intersecting. A line is formed by the collocation points of the neighbor REV. And the vector $\vec{n_{ab}}$ is normal to the boundary $\sigma_{ab}$ (admissible grid)."

# ╔═╡ 2069d5fc-2885-4d80-b51b-be5b2fccdd84
md"Finally, this leads to approximation:

$∇u.\vec{n_{ab}}≈ \frac{u_a-u_b}{|x_a-x_b|}$ 

We have collocation point $x_m \in \partial\Omega$ when the REV $w_a$ has a mutual edge with $∂Ω$ at $Γ_a$ which is named as **$γ_{am}$**. So we can put boundary values at the collocation points."

# ╔═╡ 05d63b9e-22a4-4ab6-a292-98c5477e8eef
md"Thanks to this we can assign boundary values at the collocation points."

# ╔═╡ e474cc3a-ad19-4885-81e0-2bf21d6c6f77
md"* In 1D: 

$w_a= \begin{cases}
&(x_1,\frac{x_1+x_2}{2}) & a = 1\\
&(\frac{x_{a-1}+x_a}{2},\frac{x_{a}+x_{a+1}}{2}) & 1< a < n\\
&(\frac{x_{n-1}+x_n}{2}, x_n) & a = n\\
\end{cases}\\$
"

# ╔═╡ c364bf8a-e546-4e8c-9d7f-f82b36f293af
md"* In 2D: 

$\Omega = (a,b) \times (c,d) \subset \mathbb{R}^2$

Assume the subdivisions:

$x_1=a<x_2<x_3<x_{n-1}<x_n=b$
$y_1=c<y_2<y_3<y_{n-1}<y_n=d$

Creating **$w_a^x$** and **$w_a^y$** and then set $\vec{x_{ab}}= (x_a,y_b)$ and $w_{ab}=w_a^x\times w_a^y$. 

Then we should find the collocation points: $\vec{x_{ab}}=(x_a,y_b)$ and $w_{ab}=w_a^x × w_a^y$. 

Unfortunately, this creates rectangular meshes. If we have a boundary then we can build restricted Voronoi cells $w_a$ with $\vec{x_a}\in w_a.$

Using Delaunay triangulation on collocation points $\vec{x_a}$ and by restricting Voronoi cells in a way that they are at the center of the circle connecting the midpoint of the boundary edges, we can fulfill the admissibility condition of grids."

# ╔═╡ 56694009-7a33-4b48-a903-4c9c908dff86
md"#### 3.1.5. Discretization of Second Order PDE"

# ╔═╡ 5f817d98-e8a3-4599-a205-0b3c7b5fc3aa
md"$\nabla \cdot \vec{j} = f \;\;\;\; (\mathrm{III})$  "

# ╔═╡ def52718-7c23-4516-8e2f-7e5d66aadfa5
md" In order to solve equation ($III$) we need to integrate it over control volume $w_a$. "

# ╔═╡ 236b584a-65c7-435b-8c17-c87bb27a5837
md"""

```math
\begin{aligned}
  0&=\int_{\omega_a} \nabla\cdot\vec{j} \ d\omega - \int_{\omega_a}f\ d\omega \\
   &=\int_{\partial\omega_a} \vec{j}\cdot \vec{n}_{\omega}\ ds  - \int_{\omega_a}f\ d\omega \\
   &=\sum_{b\in \mathcal{N}_a} \int_{\sigma_{ab}} \vec{j}\cdot \vec{n}_{ab}\  ds + \sum_{m\in \mathcal{G}_a} \int_{\gamma_{am}} \vec{j}\cdot \vec{n}_m \ ds
     - \int_{\omega_a}f d\omega,
\end{aligned}
```

which is equal to flux between CV $+$ flux in/out of the boundary of  $\Omega$ $-$  sources.
"""

# ╔═╡ cecfaaba-da03-48a9-a0a2-1cd1dde8291b
md" After this, we need to approximate the flux between the control volumes and also the specification of the boundary condition is required. The exact solution is the Barenblatt equation which can define the function at the boundary. The begin with the iteration scheme for time-dependent value $\vec{u}(\vec{x},t)$ we have to introduce initial values into the Finite Volume Method."

# ╔═╡ 77b3f8bc-4861-4030-872e-ee1005fc9297
md"#### **_Approximations:_**
For the simplicity, let $u_a = u(\vec{x}_a)$ and $u_b = u(\vec{x}_b)$.

* Approximation of normal derivative: 
$∇u.\vec{n}_{ab}≈ \frac{u_a-u_b}{x_a-x_b}\;\;\;\; (\mathrm{VIII})$ 

* Approximation of the flux between REV's: 
$\int_{σab}\vec{j}.\vec{n}_{ab}ds≈ \frac{|σ_{ab}|}{|x_a-x_b|}δ(u_a - u_b):= \frac{|σ_{ab}|}{|x_a - x_b|}g(u_a ,u_b)\;\;\;\; (\mathrm{IX})$

* Approximation of $\vec{j}\cdot \vec{n_m}$ at the boundary of $w_a$:
$\vec{j}.\vec{n_{m}} ≈ \alpha_mu_a-β_m$  

* Approximation of flux from $w_a$ through $\Gamma_m$: 
$\int_{γ_{am}}\vec{j}.\vec{n_{m}}.ds≈ |γ_{am}|(\alpha_mu_a-β_m)\;\;\;\; (\mathrm{X})$

* Approximation of the source function: 
$f_a=\frac{1}{w_a}\int_{w_a}f(\vec{x})dw\;\;\;\; (\mathrm{XI})$
where g is the flux function. "

# ╔═╡ 5d126701-ac46-4beb-883c-2d32770ec46a
md" The flux function can be writen in general form as $\vec{j}= -D(u) \vec{∇}u$.

Since $D(u) = mu^{m-1}$, taking the integral to approximate the function we get

* Approximation of the flux function gives us:

$D(u)= \int_0^uD(u) = u^m\;\;\;\; (\mathrm{XII})$

* So the flux function between the points a and b becomes: 

$g(u_a,u_b) = u_a^m - u_b^m\;\;\;\; (\mathrm{XIII})$    

When we solve this equation we get the $N \times N$ matrix. We should also write the Jacobi matrix of the system to find the residual at each step."

# ╔═╡ 90e4b9e2-47f1-48da-a205-69e170467add
md"""

##### Discretized system of equations
For each control volume, we get the discrete system of equations then writes for $a\in\mathcal{N}$:
```math
\begin{aligned}
  \sum_{b\in\mathcal{N}_a} \frac{|\sigma_{ab}|}{h_{ab}} \delta(u_a -  u_b)
  + \sum_{m\in \mathcal{G}_a} |\gamma_{am}| \alpha_m u_a& =  |\omega_a| f_a + \sum_{m\in \mathcal{G}_a} |\gamma_{am}|\beta_m\\
  u_a\left( \delta \sum_{b\in\mathcal{N}_a} \frac{|\sigma_{ab}|}{h_{ab}} + \alpha_m \sum_{m\in \mathcal{G}_a} |\gamma_{am}| \right)
  -  \delta \sum_{b\in\mathcal{N}_a} \frac{|\sigma_{ab}|}{h_{ab}} u_b &=  |\omega_a| f_a + \sum_{m\in \mathcal{G}_a} |\gamma_{am}|\beta_m\\
\end{aligned}
```
This can be rewritten as
```math
\begin{aligned}
    Au&=b\\
    a_{aa} u_a + \sum_{b=1\dots |\mathcal{N}|, b\neq a} a_{ab} u_b &= b_a&\text{for}\;a=1\dots |\mathcal{N}|
\end{aligned}
```
with coefficients
```math
\begin{aligned}
 a_{ab} &=
 \begin{cases}
   \sum_{b'\in \mathcal N_a} \delta\frac{|\sigma_{ab'}|}{h_{ab'}} + \sum_{m\in \mathcal{G}_a} |\gamma_{am}|\alpha_m, & b=a\\
   -\delta\frac{\sigma_{ab}}{h_{ab}}, & b\in \mathcal N_a\\
   0, & \text{else}
 \end{cases}\\
b_a&=|\omega_a| f_a + \sum_{m\in \mathcal{G}_a} |\gamma_{am}|\beta_m.
\end{aligned}
```
Now, the partial differential equation is able to turn to the linear system of the equations. Note that this matrix has ``N=|\mathcal N|`` equations (one for each control volume ``\omega_a``) and ``N=|\mathcal N|`` unknowns (one for each collocation point ``x_a\in \omega_a``).
"""

# ╔═╡ 3e33aef8-3acb-4a46-9c0a-3e010ccca1ad
md"""
##### Matrix assembly algorithm
From the relation between the Voronoi diagram and Delaunay triangulation, and from the triangulation we can assemble the discrete system. The assembly of the algorithm in two loops: Loop over all triangles, calculate triangle contribution to matrix entries, and Loop over all boundary segments, calculate contribution to matrix entries.
The piecewise solution at each REV $w_a$ is given by the solution of this matrix and the piecewise linear function on triangles is $x_a$.
"""


# ╔═╡ 103c2689-a09d-4e23-9d24-7c17caac74ae
md"### 3.2. Discretization in Time"

# ╔═╡ 4e2c7812-c899-423a-bfa0-e05722923d82
md"#### 3.2.1. Time Discretization of the Transient Problem"

# ╔═╡ ca8fd5f7-a456-4c4f-9fcd-9c186808bd13
md"To implement the time discretization of the transient problem we need to know the boundary condition, flux function, and the initial values at the boundaries. 

- We can either discretize the problem in space and after that in time (Method of Lines). By applying this method we get a huge ODE system, then we apply methods for the solution of systems of ordinary differential equations.

- Or, we can first discretize it in time and then in space (Rothe method). 

→ The difference is more or less formal."

# ╔═╡ 95da7f31-b5ad-4c02-adcf-07408ac77e0b
md"Steps of discretization: 

1. Choose time discretization points: $0 = t^0 < t^1...< t^N = T$
2. Set $τ^n = t^n - t^{n-1}$
3. Approximate the time derivative by a finite difference in time.
4. Evaluate the main part of the equation for a value interpolated between the old and the new timestep."

# ╔═╡ 9b3755d0-3ce3-451b-ba8b-aacda52b6ed0
md"For $n=1 ...N$, we need to solve: 
```math
\begin{aligned}
\frac{u^n-u^{n-1}}{τ^n}- ∇.D∇u^θ&=f\;\;\;\;\text{in} Ω×[0,T]\\
D∇u^θ.\vec{n}+ αu^θ &= g \;\;\;\;\text{on}  ∂Ω×[0,T]
\end{aligned}
```


where $u^θ = θu^n + (1-θ)u^{n-1}$. And there are three types of Euler method:

* For $θ = 1$ : Backward (implicit) Euler method: Solve PDE problem in each timestep. First-order accuracy in time.
* For $θ = \frac{1}{2}$ : Crank-Nicolson scheme: Solve PDE problem in each timestep. Second-order accuracy in time.
* For $θ = 0$ : Forward (explicit) Euler method: First order accurate in time. This does not involve the solution of a PDE problem.

"

# ╔═╡ 6d0f023e-7be5-42d9-b8fc-f3f790b549c3
md"We are going to use __VoronoiFVM.jl__ to build the FVM, create the grids, and implement the discrete system assembly. This package uses **implicit Euler method plus damped Newton method** to solve the problem."


# ╔═╡ a3fc98d0-2e29-43d4-842f-b87bf33087f7
md"#### 3.2.2. Time Discretization for a Homogeneous Neumann Problem"

# ╔═╡ d77c5bb6-0cde-4a77-a829-b14d4f6121c7
md"

We want to find a function $u: Ω × [0,T] → \mathbb{R}$ such that   $u(x,0)=u_0(x)$ and

```math
\begin{aligned}
∂_tu-∇.D∇u=0\;\;\;\;\text{in}\;\;Ω×[0,T]\\
D∇u.\vec{n}=0\;\;\;\;\text{on}\;\;Γ×[0,T]
\end{aligned}
```   

If we integrate the equation over space-time control volume $w_a × (t^{n-1},t ^n)$ divide by $τ^n$: 

$0 = \int_{w_a}((\frac{1}{τ^n})(u^n-u^{n-1})-∇.D∇u^θ)dw$
$=\frac{1}{τ^n}\int_{w_a}(u^n-u^{n-1})dw-\int_{∂w_a}D∇u^θ.\vec{n_a}dγ$
$= \sum_{l∈ \mathcal{N_a}}\int_{σ_{ab}}D∇u^θ.\vec{n_{ab}}dγ-\int_{\gamma a}D∇u^θ.\vec{n}dγ +\frac{1}{τ^n} \int_{w_a}(u^n-u^{n-1})dw$
$≈\underbrace{\frac{|w_a|}{τ^n}(u_a^n-u_a^{n-1})}+ \underbrace{\sum_{l\in \mathcal{N_a}} \frac{|\sigma_{ab}|}{h_{ab}}(u_a^\theta- u_l^{\theta})}$
$→ M          → A$

Resulting matrix equation:

$\frac{1}{τ^n}(Mu^n- Mu^{n-1}) + Au^\theta = 0$
$\frac{1}{τ^n}Mu^n + \theta Au^n =\frac{1}{τ^n} M u^{n-1}+(\theta-1)Au^{n-1}$
$u^n + \tau ^nM^{-1}\theta Au^n = u^{n-1}+\tau ^nM^{-1}(\theta-1)Au^{n-1}$
Since $M = (m_{ab})$ and $A = (\alpha_{ab})$ with:

$\alpha_{ab}=\begin{cases}
&\sum_{b'\in \mathcal{N_a}} D \frac{|\sigma_{ab'}|}{h_{ab'}} & b = a\\
&-D\frac{\sigma_{ab}}{h_{ab}} & b ∈ N_a\\
&0 & else\\
\end{cases}\\$


$m_{ab}= \begin{cases}
&|w_a| & b = a\\
&0 & else\\
\end{cases}\\$

* ⇒ $\theta A + M$ is strictly diagonally dominant.
"

# ╔═╡ 3200be1e-bae9-4381-a022-8d6eab313548
md"## 4. Solution Results"

# ╔═╡ 1900afc5-47fa-45cb-9652-18d636c9f9de
md" 1D Grid in $Ω=(-1,1)$ that is divided into N=$(@bind N Slider(2:1:30,default=30,show_value=true)) points at x axis."

# ╔═╡ 548f3fb1-6159-4cf2-878c-6a7b4f719122
begin
X1= range(-1,1;length=N);
g1 = simplexgrid(X1); @show g1;
gridplot(g1; Plotter=PyPlot, resolution=(500,125))
end

# ╔═╡ b44a8ae0-9279-432e-8b14-e4665146c710
md"2D Grid in $Ω=(-1,1) × (-1,1)$ divided into $N$ points at x and y axises."

# ╔═╡ 387bb56d-c382-4a81-9bed-9780ad0406df
X2= collect(-1:1.0/sqrt(N):1);

# ╔═╡ a1a8ec12-fd2f-4427-b0af-cc8c640cb848
begin 
	g2 = simplexgrid(X2,X2)
	cellmask!(g2, [0.0,0.0], [1,1], 3)
	bfacemask!(g2, [0.0, 0.0], [0.0, 0.5],2)
end	;

# ╔═╡ 9e5ffbe3-9f89-44b4-a62e-d1ab83b36c95
gridplot(g2,Plotter=PyPlot,resolution=(400,400))

# ╔═╡ 229e3112-2701-418f-be81-675cf4ada803
md"### 4.1. Approximation of the Solution of Porous Medium Equation by using Neumann Boundary Condition"

# ╔═╡ 3780fdb0-b4fb-44bf-bd70-665bfeac3a33
md"#### 4.1.1. Approximation in 1D and 2D"

# ╔═╡ 5b6cdfc5-3db7-4c98-ab71-2d38552f0cb7
function iter(inval, sys, grid, tstep, tend, dtgrowth)
	time=0.001
	times=[time]
	solutions=[copy(inval)]
	solution=copy(inval)
	while time<tend
	 	time+=tstep
		VoronoiFVM.solve!(solution,inval,sys,tstep=tstep)#implicit Euler timestep
		inval.=solution # copy solution to inivalue
		push!(times,time)
		push!(solutions,copy(solution))
		tstep*=dtgrowth
	end
     (times=times,
	solutions=solutions,
	grid=grid)
end

# ╔═╡ 97652feb-4b21-4665-8dca-571d476d2320
md"""
Value of m=$(@bind m Slider(2:1:10,default=2, show_value=true)) in Barenblatt.
"""

# ╔═╡ f2583a51-7c2f-4462-99b6-a1832508f721
function diffusion(;tstep=0.0001,tend=0.1,dtgrowth=1.1,DiffEq=nothing)
	grid = g1
	function flux!(f,u0,edge)
        u=unknowns(edge,u0)
        f[1]=u[1,1]^m-u[1,2]^m   # Flux function
	end
	## Storage term (under time derivative)
	function storage!(f,u,node)  # Source function
		f[1]=u[1]
	end
	## Create a physics structure
	physics=VoronoiFVM.Physics(flux=flux!, storage=storage!)
	sys=VoronoiFVM.DenseSystem(grid,physics)
	enable_species!(sys,1,[1])
	
	## Create a solution array
	inval=unknowns(sys)
	
	## Boundary condition
	fbound(x)=exp(-100*(x^2))
	
	## Broadcast the initial value
	inval[1,:].=map(fbound,grid)
	iter(inval,sys,grid,tstep,tend,dtgrowth)
end

# ╔═╡ 13ea4cc6-ffcf-4b9a-a236-e2ec50f293bd
solution_NB=diffusion();

# ╔═╡ 6df575c4-e5e5-4ca0-9df9-b6c3048ecd8c
function diffusion2d(;tstep=0.0001,tend=0.1,dtgrowth=1.1,DiffEq=nothing)
    g2
	function flux!(f,u0,edge)
        u=unknowns(edge,u0)
        f[1]=u[1,1]^m-u[1,2]^m    # Flux function
	end
	function storage!(f,u,node)   # Source function
		f[1]=u[1]
	end
	
	## Create a physics structure
	physics=VoronoiFVM.Physics(flux=flux!, storage=storage!) 
	sys=VoronoiFVM.DenseSystem(g2,physics)
	enable_species!(sys,1,[1])
	
	## Create a solution array
	inval=unknowns(sys)

	## Boundary condition
	fbound2D(x,y)=exp(-100*((x)^2+(y)^2))

	## Broadcast the initial value
	inval[1,:].=map(fbound2D,g2)
	iter(inval,sys,g2,tstep,tend,dtgrowth)
end

# ╔═╡ b21f2230-b2f5-4cd7-b001-f833fe9ce8e2
solution_NB_2D=diffusion2d();

# ╔═╡ 99dd0219-069e-498a-abdf-b359ae868ea3
md" Plot time space solutions in 1D and 2D Grids: "

# ╔═╡ e5fde0eb-c15d-400b-a0d0-4e5157fd618e
md"""
time=$(@bind t_diffusion Slider(1:length(solution_NB.times),default=1,show_value=true))
"""

# ╔═╡ 1623c8b9-cb84-4839-8e4e-fbbd7644c215
begin
	ps=GridVisualizer(Plotter=PyPlot,fast=true,layout =(1,1))
	scalarplot!(ps[1,1],g1,solution_NB.solutions[t_diffusion][1,:],title="Figure 1. Space time solution 1D using Neumann BC at time=$(solution_NB.times[t_diffusion])",label="Solution by using VoronoiFVM",resolution=(600,400),markershape=:hexagon,markevery=2,color =:blue,clear=true)
	legend()
	PyPlot.grid()
	gcf().set_size_inches(10,3)
	gcf()
end

# ╔═╡ cfc0dc88-f8ba-4c44-901f-cb3d3a6bfe27
scalarplot(solution_NB_2D.grid,solution_NB_2D.solutions[t_diffusion][1,:],title="Figure 2. Space time solution 2D using Neumann BC

t=$(solution_NB_2D.times[t_diffusion])",Plotter=PyPlot,resolution=(600,570),isolines=10,colormap=:cool,levels=40)

# ╔═╡ f272e9e7-0084-4ac8-a675-feae3b34d6bd
md"### 4.2. Approximation of the Solution of Porous Medium Equation by Improved Solution" 

# ╔═╡ 63e87785-64d5-4dec-be38-b8d20d22b4ca
md"We are using Barenblatt solution to give the exact values at the boundaries. As a result we get improved numerical solution which has less error than the result that we got by using Neumann Boundary condition. "

# ╔═╡ 93b0d805-afb8-49b1-898e-e400d1dee96c
md"#### 4.2.1. Approximation in 1D and 2D"

# ╔═╡ 2021c063-a562-4333-8f4e-a55c18eb3dec
function barenblatt(x,t,m,d=1)
	tx=t^(-1.0/(m+1.0))
    xx=x*tx
    xx=xx*xx
    xx=1- xx*(m-1)/(2.0*m*(m+1));
    if xx<0.0
        xx=0.0
    end
    return tx*xx^(1.0/(m-1.0))
end

# ╔═╡ ffb60616-6ef3-4af0-a61a-c412d23b2092
function barenblatt2d(x,y,t,m,dim=2)
	
	alpha=(1/(m-1+(2/dim)))
	r=sqrt(x^2+y^2)
	K=(alpha*(m-1)*r^2)/(2*dim*m*(t^(2*alpha/dim)))
	
	B=(t^(-alpha))*((1-K)^(1/(m-1)));
	
    if B<0.0
        B=0.0
    end
	
    return B
end

# ╔═╡ 8d2f6103-4035-415a-8000-37f53201bd9f
function diffusion_with_br_1d(;tstep=1.0e-4,tend=0.1, dtgrowth=1.1)
	
	function simple_grid(N)
	h=1.0/convert(Float64,N)
	X=collect(-1.0:h:1.0)       #according to domain when L=1
	g1_k= VoronoiFVM.Grid(X)    #simplexgrid(X)
 	end
	grid=simple_grid(N)	
	#flux 
	function flux!(f,u0,edge)  
        u=unknowns(edge,u0)
        f[1]=u[1,1]^m-u[1,2]^m
	end
		
	#storage function
	function storage!(f,u,node)  
		f[1]=u[1]
	end	
	
	#create physics	
	physics=VoronoiFVM.Physics(
	flux=flux!,
	storage=storage!)   
		
	#create system	
	sys=VoronoiFVM.DenseSystem(grid,physics)   
	enable_species!(sys,1,[1])
	
	#initial value	
	inval=unknowns(sys)
	t0=0.0001
		
	# Broadcast the initial value
	inval[1,:].=map(x->barenblatt(x,t0,m),grid)
		
	# Create solver control info for constant time step size	
	control=VoronoiFVM.NewtonControl()
	control.Δt_min=0.01*tstep
	control.Δt=tstep
	control.Δt_max=0.1*tend
	control.Δu_opt=1
	control.Δt_grow=dtgrowth
	
	
	tsol=VoronoiFVM.solve(inval,sys,[t0,tend],control=control)
     (grid=grid,
		solution=tsol)
end

# ╔═╡ 4e48b49e-1963-4c53-aa16-bc124b4b8235
md"""##### Choosing the Time Values $t_0 < t < t_1$ so that the Boundary Value b(x, $t$) is contained in the Domain $(-1,1)$
"""

# ╔═╡ 43f6ddd5-8e4d-4dc8-8fb7-983033c240bc
md" The Barenblatt solution

$u(x,t)= max(0,t^{-\alpha}(1-\frac{\alpha(m-1)r^{2}}{2dmt^{\frac{2\alpha}{d}}})^{\frac{1}{m-1}})$  

where $r=|x|$ and $\alpha=\frac{1}{m-1+\frac{2}{d}}$, doesn't leave the domain if $\;\;$ $1.0e^{-5}<t<0.01$."

# ╔═╡ b6a45689-4bdf-4263-a229-0b7e38e5cac2
md"""
t=$(@bind t1 Slider(0.00000001:0.00001:0.01,default=0.0001,show_value=true))
"""

# ╔═╡ 660c4ab9-14ac-4604-b60f-b999c027a2b7
let
	clf()
	PyPlot.grid()
	X=collect(range(-1,1,length=N))
	PyPlot.plot(X,map(x->barenblatt(x,10^(-2),m),X),label=@sprintf("upper time, t_1=%.3g",10^(-2)))
	PyPlot.plot(X,map(x->barenblatt(x,t1,m),X), label="t")
	PyPlot.plot(X,map(x->barenblatt(x,10^(-5),m),X),label =@sprintf("lower time, t_0=%.3g",10^(-5)))
	PyPlot.xlabel("grid in the domain (-1,1)")
	PyPlot.ylabel("exact solution")
	PyPlot.title("Figure 3. Space time solution 1D using Barenblatt Solution")
	legend()
	gcf().set_size_inches(10,3)
	gcf()

end

# ╔═╡ 29176cf7-dcd6-48c3-bfb6-056ea44ed1f5
P_grid,P_tsol = diffusion_with_br_1d();

# ╔═╡ b02abf85-4a48-4643-aea6-cee18e1fb0d4
md"""
t :$(@bind t_b Slider(1:length(P_tsol),default=1,show_value=true))
"""

# ╔═╡ 1a20824c-db94-4116-8bc0-d2fb9a736821
begin
			p1=GridVisualizer(Plotter=PyPlot,layout=(1,1),fast=true,resolution=(1000,1000),colormap=:cool,levels=50)
	       	scalarplot!(p1[1,1],P_grid,P_tsol[1,:,t_b],title="Figure 4. Improved Numerical solution of Neumann boundary condition in 1D",color=:blue,clear=true,label=@sprintf("Numerical solution, t=%.3g",P_tsol.t[t_b]),markershape=:circle,markevery=1,smooth=true)
	PyPlot.grid()
			reveal(p1)
			legend()
		PyPlot.grid()
		gcf().set_size_inches(10,3)
		gcf()
end

# ╔═╡ 3e5e5803-6e9a-4860-a59c-bdfb56fd4973
md"### 4.3. Comparison and Error Estimation "

# ╔═╡ 993685c3-7244-454c-91da-f4e013920fe7
md"
#### 1D case
In order to estimate which approximation converges to the exact solution more closely we need to compare the solutions of Neumann Boundary condition and Barenblatt solution. In addition, we need to estimate the error. "

# ╔═╡ ba2fd3d2-28da-4cb2-85fa-cf7630d1aef6
md"""
t :$(@bind t_b1 Slider(1:length(P_tsol),default=20,show_value=true))
"""

# ╔═╡ bed0d657-b10f-427e-93f8-0c1cf71c6a76
begin
			p3=GridVisualizer(Plotter=PyPlot,layout=(1,1),fast=true,resolution=(1000,1000),colormap=:cool,levels=50)
	       	scalarplot!(p3[1,1],P_grid,P_tsol[1,:,t_b1],title="Figure 6. Space Time Solution",color=:blue,clear=true,label=@sprintf("Improved Numerical Solution, t=%.3g",P_tsol.t[t_b1]),markershape=:circle,markevery=1,smooth=true)
			scalarplot!(p3[1,1],P_grid,map(x->barenblatt(x,P_tsol.t[t_b1],m),P_grid),color=:red,levels=50,label=@sprintf("Exact Solution i.e barenblatt, t=%.3g",P_tsol.t[t_b1]), markershape=:hexagon,markevery=3,clear=false)
			PyPlot.grid()
			reveal(p3)
			legend()
		PyPlot.grid()
		gcf().set_size_inches(10,3)
		gcf()
end

# ╔═╡ 201993e6-baac-40f0-9f07-f55e19cad371
begin
	p6=GridVisualizer(Plotter=PyPlot,layout=(1,1),fast=true,resolution=(1000,1000),colormap=:cool,levels=50)
		
	scalarplot!(p6[1,1],P_grid, P_tsol[1,:,t_b1]-map(x->barenblatt(x,P_tsol.t[t_b1],m),P_grid),title="Figure 7. The error between numerical solution and exact solution",color=:black,clear=true,label=@sprintf("Numerical Solution with Neumann and Barenblatt Solution, t=%.3g",P_tsol.t[t_b1]),markershape=:circle,markevery=1,smooth=true)
		PyPlot.grid()
	PyPlot.grid()
	gcf().set_size_inches(10,3)
	gcf()
end


# ╔═╡ 603bb8de-b7e1-4bf7-9001-40017508eece
begin
	A=[]
	for i = 1:length(P_tsol[1,:])
		append!(A,sum(abs.(P_tsol[1,:,i]))/length(P_tsol[1,:,i]))
	end
	A
end;


# ╔═╡ f2925f54-7382-4ba7-85ee-f57bc9de57d1
md"
Since the error is getting bigger as time pass, the sum of the absolute value of error is plotted with respect to the time.
"

# ╔═╡ 0b95ac76-9075-4854-a78d-24ea07806dfa
begin
	p9=GridVisualizer(Plotter=PyPlot,layout=(1,1),fast=true,resolution=(1000,1000),colormap=:cool,levels=50)
		
	scalarplot!(p9[1,1],P_tsol.t, A,title="Figure 8. Sum of the absolute value of error at each time",color=:green,clear=true,markershape=:circle,markevery=1,smooth=true)
		PyPlot.grid()
	xlabel("t")
	ylabel("sum of the absolute value")
	PyPlot.grid()
	gcf().set_size_inches(10,3)
	gcf()
end


# ╔═╡ 11cc06a2-f93b-4271-ad29-cbd1124925af
md"
This graph shows that the sum of the absolute value of error is getting bigger after at some time point. Moreover, Figure 9. shows that the numerical solution using only the Neumann boundary conditions does not give a good approximation.
"

# ╔═╡ 01ea84dc-7f24-4c45-b4ef-de3b4283af87
begin
			p4=GridVisualizer(Plotter=PyPlot,layout=(1,1),fast=true,resolution=(1000,1000),colormap=:cool,levels=50)
	
	       	scalarplot!(p4[1,1],g1,solution_NB.solutions[t_diffusion][1,:],title="Figure 9. Space Time Solution",label="Numerical Solution with Neumann BC",resolution=(600,400),markershape=:hexagon,markevery=2,color =:blue,clear=true)
	
			scalarplot!(p4[1,1],P_grid,map(x->barenblatt(x,P_tsol.t[t_b1],m),P_grid),color=:red,levels=50,label=@sprintf("Exact Solution i.e barenblatt, t=%.3g",P_tsol.t[t_b1]), markershape=:hexagon,markevery=3,clear=false)

			PyPlot.grid()
			reveal(p4)
			legend()
		PyPlot.grid()
		gcf().set_size_inches(10,3)
		gcf()
end

# ╔═╡ d0b4aef3-0d4b-4370-a059-af28a31b1955
md"
Hence, we conclude that the proper boundary point is crucial in approximation of the numerical solution.
"

# ╔═╡ a1a353df-a89b-466e-b0c7-b07db96c2756
# Create discretization grid in 1D or 2D with approximately n nodes
function create_grid(n,dim)
	nx=n
	if dim==2
		nx=ceil(sqrt(n))
	end
	X=collect(-1:1.0/nx:1)
	if dim==1
      grid=simplexgrid(X)
	else
      grid=simplexgrid(X,X)
	end
end;

# ╔═╡ cc2993ca-7b78-406a-bcd0-068bc9f8f200
function diffusion_with_br_2d(;n=100,dim=2,tstep=1.0e-4,tend=0.01, dtgrowth=1.1)
	# h=1.0/convert(Float64,n)
	# X=collect(-1.0:h:1.0)
	# Y=collect(-1.0:h:1.0)
	# grid = VoronoiFVM.Grid(X,Y)
	grid=create_grid(n,dim)

	#flux 
	function flux!(f,u0,edge)  
        u=unknowns(edge,u0)
        f[1]=u[1,1]^m-u[1,2]^m
	end
		
	#storage
	function storage!(f,u,node)  
		f[1]=u[1]
	end
		
	#create physics	
	physics=VoronoiFVM.Physics(
	flux=flux!,
	storage=storage!)   
		
	#create system	
	sys2D=VoronoiFVM.DenseSystem(grid,physics)   
	enable_species!(sys2D,1,[1])
	
	#initial value	
	inval=unknowns(sys2D)
	t0=0.001
	
	# Broadcast the initial value
	inval[1,:].=map((x,y)->barenblatt2d(x,y,t0,m),grid)
		
	# Create solver control info for constant time step size	
	control=VoronoiFVM.NewtonControl()
	control.Δt_min=0.01*tstep
	control.Δt=tstep
	control.Δt_max=0.1*tend
	control.Δu_opt=1
	control.Δt_grow=dtgrowth
	
	tsol=VoronoiFVM.solve(inval,sys2D,[t0,tend],control=control)
	
	return [grid,tsol[1:end]]
end

# ╔═╡ d8c3f5fb-907d-47d3-9be6-b536f9326599
tsol_grid,tsol2d= diffusion_with_br_2d(dim=2,n=N);

# ╔═╡ 4f783416-3603-401d-aed4-f8d53501691a
md"""
t :$(@bind t2d Slider(1:length(tsol2d),default=29,show_value=true))
"""

# ╔═╡ 04ffb19b-eefb-4c11-a664-ebb89fd1670f
begin
	p2=GridVisualizer(Plotter=PyPlot,layout=(1,1),fast=true,resolution=(600,600),colormap=:cool,levels=50)

	       scalarplot!(p2[1,1],tsol_grid,tsol2d[1,:,t2d],title=@sprintf("Figure 5. Improved Numerical solution of Neumann boundary condition in 2D, t=%.3g",tsol2d.t[t2d]))

gcf().set_size_inches(7,7)
		gcf()
	end

# ╔═╡ 0b035086-013c-4884-82b4-59a8425cc0af
md"""
Time step number:$(@bind t_b2d Slider(1:length(tsol2d),default=10,show_value=true))
"""

# ╔═╡ 67a34131-abbf-44a6-9a83-0a88ed9dbf47
function porous_diffusion2D(;n=100,dim=2,tstep=1.0e-4,tend=0.01, dtgrowth=1.1)
	grid=create_grid(n,dim)

		
	#flux 
	function flux!(f,u0,edge)  
        u=unknowns(edge,u0)
        f[1]=u[1,1]^m-u[1,2]^m
	end
		
	#storage
	function storage!(f,u,node)  
		f[1]=u[1]
	end
		
	
	
		
	#create physics	
	physics=VoronoiFVM.Physics(
	flux=flux!,
	storage=storage!)   
		
	#create system	
	sys2D=VoronoiFVM.DenseSystem(grid,physics)   
	enable_species!(sys2D,1,[1])
	
	#initial value	
	inival=unknowns(sys2D)
	t0=0.001
	
   
	# Broadcast the initial value
	inival[1,:].=map((x,y)->barenblatt2d(x,y,t0,m),grid)
		
	# Create solver control info for constant time step size	
	control=VoronoiFVM.NewtonControl()
	control.Δt_min=0.01*tstep
	control.Δt=tstep
	control.Δt_max=0.1*tend
	control.Δu_opt=1
	control.Δt_grow=dtgrowth
	
	
	tsol=solve(inival,sys2D,[t0,tend],control=control)
	
	return [grid,tsol[1:end]]
end

# ╔═╡ 06203d38-d9a7-489e-a655-186187e6c9a0
md"
#### 2D case
"

# ╔═╡ bc5be9f6-ee45-4641-b26b-295ce4f0aca7
let
	clf()	
	suptitle("Figure 10. Improved Numerical solution of Neumann boundary condition in 2D")
		
	surf(tsol_grid[Coordinates][1,:],tsol_grid[Coordinates][2,:], tsol2d[1,:,t_b2d],cmap=:summer) # 3D surface plot
	ax=gca(projection="3d")  # Obtain 3D plot axes
	
	xlabel("x")
	ylabel("y")
	gcf()
	
end

# ╔═╡ 302ae7c1-b9e3-4a11-9241-d5db87e392a6
let
	clf()
	
	suptitle("Figure 11. Numerical Solution with Neumann b.c.")
	surf(solution_NB_2D.grid[Coordinates][1,:],solution_NB_2D.grid[Coordinates][2,:], solution_NB_2D.solutions[1][:],cmap=:autumn
) # 3D surface plot
	ax=gca(projection="3d")
	xlabel("x")
	ylabel("y")
	gcf()
end

# ╔═╡ 380e21e7-17f8-41f6-ac68-da8be570888c
let
	clf()
	
	suptitle("Figure 12. Exact Barenblatt Solution")
	surf(tsol_grid[Coordinates][1,:],tsol_grid[Coordinates][2,:], map((x,y)->barenblatt2d(x,y,tsol2d.t[t_b2d],m),tsol_grid),cmap=:winter) # 3D surface plot
	ax=gca(projection="3d")
	xlabel("x")
	ylabel("y")
	gcf()
end

# ╔═╡ 53a4ec18-2f8f-4d35-b01c-b07488088856
let
	clf()
	
	suptitle("Figure 13. Error of the Numerical Solution")
	surf(tsol_grid[Coordinates][1,:],tsol_grid[Coordinates][2,:], map((x,y)->barenblatt2d(x,y,tsol2d.t[t_b2d],m),tsol_grid)-tsol2d[1,:,t_b2d],cmap=:spring) # 3D surface plot
	ax=gca(projection="3d")
	xlabel("x")
	ylabel("y")
	gcf()
end

# ╔═╡ 3a356aa0-b4d0-4400-a610-47510a8a4d0b
md"## 5. Conclusion

In our project, we consider the porous medium equation with $m > 1$, and solve the equation for $m=2$. We use the Barenblatt solution to get the exact boundary value using the Neumann boundary condition. We discretize the domain to $N$ points, then we solve the Porous medium equation by implementing **VoronoiFVM**. In the end, we compare 1D and 2D solutions with the exact solutions to estimate the error. 

#### _Remarks:_
* Choosing Boundary values is crucial in gas transport in the porous medium equation since as we observe in Figure 6 and Figure 9, the errors increase with improper (not exact) boundary values.
* From Figure 8. we observe that the error increase with respect to time after time reaches a certain point.
* __VoronoiFVM.jl__ is a powerful package for solving nonlinear partial differential equations based on the Voronoi finite volume method.

"

# ╔═╡ 4db45dfa-045b-4b59-ace7-6ddee0f375f4
md"
## 6. Reference
1. [Lecture notes on Scientific Computing WiSe 2021/22, Prof. Fruhmann](https://www.wias-berlin.de/people/fuhrmann/SciComp-WS2122/)


2. [VoronoiFVM.jl](https://j-fu.github.io/VoronoiFVM.jl/stable/)


3. [ExtendableGrids.jl](https://j-fu.github.io/ExtendableGrids.jl/dev/)


4. [Microfluidics: Modelling, Mechanics and Mathematics] (https://www.sciencedirect.com/topics/engineering/finite-volume-method)


5. [Flow in porous media I: A theoretical derivation of Darcy's law](https://link.springer.com/article/10.1007/BF01036523)
"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
ExtendableGrids = "cfc395e8-590f-11e8-1f13-43a2532b2fa8"
ExtendableSparse = "95c220a8-a1cf-11e9-0c77-dbfce5f500b3"
GridVisualize = "5eed8a63-0fb0-45eb-886d-8d5a387d12b8"
Interact = "c601a237-2ae4-5e1e-952c-7a85b0c7eef1"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PlutoVista = "646e1f28-b900-46d7-9d87-d554eb38a413"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
SimplexGridFactory = "57bfcd06-606e-45d6-baf4-4ba06da0efd5"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Triangulate = "f7e6ffb2-c36d-4f8f-a77e-16e897189344"
VoronoiFVM = "82b139dc-5afc-11e9-35da-9b9bdfd336f3"

[compat]
Colors = "~0.12.8"
DataFrames = "~1.3.3"
ExtendableGrids = "~0.8.11"
ExtendableSparse = "~0.6.7"
GridVisualize = "~0.4.7"
Interact = "~0.10.4"
Plots = "~1.27.6"
PlutoUI = "~0.7.38"
PlutoVista = "~0.8.13"
PyPlot = "~2.10.0"
SimplexGridFactory = "~0.5.13"
Triangulate = "~2.1.3"
VoronoiFVM = "~0.14.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "c933ce606f6535a7c7b98e1d86d5d1014f730596"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "5.0.7"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AssetRegistry]]
deps = ["Distributed", "JSON", "Pidfile", "SHA", "Test"]
git-tree-sha1 = "b25e88db7944f98789130d7b503276bc34bc098e"
uuid = "bf4720bc-e11a-5d0c-854e-bdca1663c893"
version = "0.1.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CSSUtil]]
deps = ["Colors", "JSON", "Markdown", "Measures", "WebIO"]
git-tree-sha1 = "b9fb4b464ec10e860abe251b91d4d049934f7399"
uuid = "70588ee8-6100-5070-97c1-3cb50ed05fe8"
version = "0.1.1"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[Cassette]]
git-tree-sha1 = "063b2e77c5537a548c5bf2f44161f1d3e1ab3227"
uuid = "7057c7e9-c182-5462-911a-8362d720325c"
version = "0.3.10"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "6e47d11ea2776bc5627421d59cdcc1296c058071"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.7.0"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "6c19003824cbebd804a51211fd3bbd81bf1ecad5"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.3"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "dd933c4ef7b4c270aacd4eb88fa64c147492acf0"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.10.0"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[ElasticArrays]]
deps = ["Adapt"]
git-tree-sha1 = "a0fcc1bb3c9ceaf07e1d0529c9806ce94be6adf9"
uuid = "fdbdab4c-e67f-52f5-8c3f-e7b388dad3d4"
version = "1.2.9"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[ExtendableGrids]]
deps = ["AbstractTrees", "Dates", "DocStringExtensions", "ElasticArrays", "InteractiveUtils", "LinearAlgebra", "Printf", "Random", "SparseArrays", "Test"]
git-tree-sha1 = "fbb0efd29f2ba5e25eeaf73b76257acfc1a28630"
uuid = "cfc395e8-590f-11e8-1f13-43a2532b2fa8"
version = "0.8.11"

[[ExtendableSparse]]
deps = ["DocStringExtensions", "LinearAlgebra", "Printf", "Requires", "SparseArrays", "SuiteSparse", "Test"]
git-tree-sha1 = "eb3393e4de326349a4b5bccd9b17ed1029a2d0ca"
uuid = "95c220a8-a1cf-11e9-0c77-dbfce5f500b3"
version = "0.6.7"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "80ced645013a5dbdc52cf70329399c35ce007fae"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.13.0"

[[FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "56956d1e4c1221000b7781104c58c34019792951"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.11.0"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "40d1546a45abd63490569695a86a2d93c2021e54"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.26"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[FunctionalCollections]]
deps = ["Test"]
git-tree-sha1 = "04cb9cfaa6ba5311973994fe3496ddec19b6292a"
uuid = "de31a74c-ac4f-5751-b3fd-e18cd04993ca"
version = "0.5.0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "af237c08bda486b74318c8070adb96efa6952530"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.2"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "cd6efcf9dc746b06709df14e462f0a3fe0786b1e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.2+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "57c021de207e234108a6f1454003120a1bf350c4"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.6.0"

[[GridVisualize]]
deps = ["ColorSchemes", "Colors", "DocStringExtensions", "ElasticArrays", "ExtendableGrids", "GeometryBasics", "HypertextLiteral", "LinearAlgebra", "Observables", "OrderedCollections", "PkgVersion", "Printf", "StaticArrays"]
git-tree-sha1 = "a16fc5b8699afedb37aacbcf71d45eb794b589ea"
uuid = "5eed8a63-0fb0-45eb-886d-8d5a387d12b8"
version = "0.4.7"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[Interact]]
deps = ["CSSUtil", "InteractBase", "JSON", "Knockout", "Observables", "OrderedCollections", "Reexport", "WebIO", "Widgets"]
git-tree-sha1 = "311f9130aeb50ac93a12dd076fa02c9a430be525"
uuid = "c601a237-2ae4-5e1e-952c-7a85b0c7eef1"
version = "0.10.4"

[[InteractBase]]
deps = ["Base64", "CSSUtil", "Colors", "Dates", "JSExpr", "JSON", "Knockout", "Observables", "OrderedCollections", "Random", "WebIO", "Widgets"]
git-tree-sha1 = "3ace4760fab1c9700ec9c68ab0e36e0856f05556"
uuid = "d3863d7c-f0c8-5437-a7b4-3ae773c01009"
version = "0.10.8"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "81b9477b49402b47fbe7f7ae0b252077f53e4a08"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.22"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[JSExpr]]
deps = ["JSON", "MacroTools", "Observables", "WebIO"]
git-tree-sha1 = "bd6c034156b1e7295450a219c4340e32e50b08b1"
uuid = "97c1335a-c9c5-57fe-bc5d-ec35cebe8660"
version = "0.5.3"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[Knockout]]
deps = ["JSExpr", "JSON", "Observables", "Test", "WebIO"]
git-tree-sha1 = "deb74017e1061d76050ff68d219217413be4ef59"
uuid = "bcebb21b-c2e3-54f8-a781-646b90f6d2cc"
version = "0.2.5"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "6f14549f7760d84b2db7a9b10b88cd3cc3025730"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.14"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a970d55c2ad8084ca317a4658ba6ce99b7523571"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.12"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "3b429f37de37f1fc603cc1de4a799dc7fbe4c0b6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.0"

[[Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "6f2dd1cf7a4bbf4f305a0d8750e351cb46dfbe80"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.27.6"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "670e559e5c8e191ded66fa9ea89c97f10376bb4c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.38"

[[PlutoVista]]
deps = ["ColorSchemes", "Colors", "DocStringExtensions", "GridVisualize", "HypertextLiteral", "UUIDs"]
git-tree-sha1 = "118d1871e3511131bae2196e238d0054bd9a62b0"
uuid = "646e1f28-b900-46d7-9d87-d554eb38a413"
version = "0.8.13"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "28ef6c7ce353f0b35d0df0d5930e0d072c1f5b9b"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.1"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "1fc929f47d7c151c839c5fc1375929766fb8edcc"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.93.1"

[[PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "14c1b795b9d764e1784713941e787e1384268103"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.10.0"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "ChainRulesCore", "DocStringExtensions", "FillArrays", "LinearAlgebra", "RecipesBase", "Requires", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "bfe14f127f3e7def02a6c2b1940b39d0dabaa3ef"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.26.3"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[SimplexGridFactory]]
deps = ["DocStringExtensions", "ElasticArrays", "ExtendableGrids", "GridVisualize", "LinearAlgebra", "Printf", "Test"]
git-tree-sha1 = "88dce0e331178bcd812d87247b89f18fabb24ddc"
uuid = "57bfcd06-606e-45d6-baf4-4ba06da0efd5"
version = "0.5.13"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SparseDiffTools]]
deps = ["Adapt", "ArrayInterface", "Compat", "DataStructures", "FiniteDiff", "ForwardDiff", "Graphs", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays", "VertexSafeGraphs"]
git-tree-sha1 = "314a07e191ea4a5ea5a2f9d6b39f03833bde5e08"
uuid = "47a9eef4-7e08-11e9-0b38-333d64bd3804"
version = "1.21.0"

[[SparsityDetection]]
deps = ["Cassette", "DocStringExtensions", "LinearAlgebra", "SparseArrays", "SpecialFunctions"]
git-tree-sha1 = "9e182a311d169cb9fe0c6501aa252983215fe692"
uuid = "684fba80-ace3-11e9-3d08-3bc7ed6f96df"
version = "0.3.4"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "cbf21db885f478e4bd73b286af6e67d1beeebe4c"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.4"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "87e9954dfa33fd145694e42337bdd3d5b07021a6"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.6.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "cd56bf18ed715e8b09f06ef8c6b781e6cdc49911"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.4"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c82aaa13b44ea00134f8c9c89819477bd3986ecd"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.3.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "57617b34fa34f91d536eb265df67c2d4519b8b98"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.5"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[Triangle_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bfdd9ef1004eb9d407af935a6f36a4e0af711369"
uuid = "5639c1d2-226c-5e70-8d55-b3095415a16a"
version = "1.6.1+0"

[[Triangulate]]
deps = ["DocStringExtensions", "Libdl", "Printf", "Test", "Triangle_jll"]
git-tree-sha1 = "796a9c0b02a3414af6065098bb7cf0e88dfa450e"
uuid = "f7e6ffb2-c36d-4f8f-a77e-16e897189344"
version = "2.1.3"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[VertexSafeGraphs]]
deps = ["Graphs"]
git-tree-sha1 = "8351f8d73d7e880bfc042a8b6922684ebeafb35c"
uuid = "19fa3120-7c27-5ec5-8db8-b0b0aa330d6f"
version = "0.2.0"

[[VoronoiFVM]]
deps = ["DiffResults", "DocStringExtensions", "ExtendableGrids", "ExtendableSparse", "ForwardDiff", "GridVisualize", "IterativeSolvers", "JLD2", "LinearAlgebra", "Printf", "RecursiveArrayTools", "SparseArrays", "SparseDiffTools", "SparsityDetection", "StaticArrays", "SuiteSparse", "Test"]
git-tree-sha1 = "0cf39adeb43883e0d33bde1c5a2afdbd34cbf330"
uuid = "82b139dc-5afc-11e9-35da-9b9bdfd336f3"
version = "0.14.0"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[WebIO]]
deps = ["AssetRegistry", "Base64", "Distributed", "FunctionalCollections", "JSON", "Logging", "Observables", "Pkg", "Random", "Requires", "Sockets", "UUIDs", "WebSockets", "Widgets"]
git-tree-sha1 = "c9529be473e97fa0b3b2642cdafcd0896b4c9494"
uuid = "0f1e0344-ec1d-5b48-a673-e5cf874b6c29"
version = "0.8.17"

[[WebSockets]]
deps = ["Base64", "Dates", "HTTP", "Logging", "Sockets"]
git-tree-sha1 = "f91a602e25fe6b89afc93cf02a4ae18ee9384ce3"
uuid = "104b5d7c-a370-577a-8038-80a2059c5097"
version = "1.5.9"

[[Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "505c31f585405fc375d99d02588f6ceaba791241"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.5"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═7a7c2ec0-c57f-11ec-12f5-c776dc208f4d
# ╟─d4467ceb-4bdb-4290-a87a-7aa83df9b8d5
# ╠═b917dad1-c4ef-44ea-a8a8-f40fadc5da1b
# ╟─1b99ce43-d892-47a3-ad1d-7452aa7f29b7
# ╟─8ac82a3f-8a18-4a94-b92c-ea9bd35c892f
# ╠═9674b832-b4a5-48a7-a759-cd65090dc7d5
# ╟─50881738-5731-43e1-b9d1-8b368c1b9ce7
# ╟─427e0684-f4b2-491a-aed0-e5488b25279d
# ╟─ba49d380-d648-4ac7-bdbe-bd5a3fdb11a4
# ╟─fa0db37f-dddf-49db-9d9d-7108461a2620
# ╟─09ef294d-3885-44db-8bbf-af065b86c40c
# ╟─0f014d9e-048a-4baa-a1d3-a6c44cbdedc8
# ╟─37d357c6-2e0a-4ac5-b987-a916ab188074
# ╟─f9153d35-5189-40f9-8761-f113687eac36
# ╟─e394b22c-299a-406f-b14a-accd0467c3ec
# ╟─c5401ec5-60e0-4b74-a7d9-bb9f613c720f
# ╟─141006b1-c91d-410b-946b-20253d90e9a5
# ╟─0339e9e2-4f55-4d93-ba24-5db761fff285
# ╟─0c64f543-cc13-4b7c-aaf1-182ff3934d3b
# ╟─58377172-d60b-47ed-b71c-c80775aee2e8
# ╟─14c7cbde-1dc0-42c6-871e-6ca70b538b77
# ╟─5107f984-5200-4e8f-a37b-293492901df9
# ╟─214da77c-e799-49d6-9bbe-3b8d59ed7d94
# ╟─82b09d1e-aff1-470b-a63c-c73a78981488
# ╟─3acc2c69-ea0b-453b-8c03-71269f07043e
# ╟─7eb263d2-76c8-4666-b6ee-b15751a87527
# ╟─1dde31c5-76f1-4a9f-91d7-70f881d91344
# ╟─5b179258-f50f-4d17-a6da-8ecd0971dba4
# ╟─ce8d5a51-8cc7-40eb-a28e-5fc8b8dbf74a
# ╟─237e76fd-da0a-45fd-bf9d-27672111a709
# ╟─efd01d2f-cd90-4b3b-88d4-50f970f9fdaa
# ╟─68137fb4-0a6b-4ff5-bbbc-6cc7a15ef488
# ╟─135757e2-7f08-424c-b448-8913cd7f7e4e
# ╟─85cf5f67-3127-4141-ad0f-07c2d86bd6ac
# ╟─91be1132-faa6-4597-a875-0efe8a706499
# ╟─775ad737-7022-4545-abb5-5d8c03818088
# ╟─2069d5fc-2885-4d80-b51b-be5b2fccdd84
# ╟─05d63b9e-22a4-4ab6-a292-98c5477e8eef
# ╟─e474cc3a-ad19-4885-81e0-2bf21d6c6f77
# ╟─c364bf8a-e546-4e8c-9d7f-f82b36f293af
# ╟─56694009-7a33-4b48-a903-4c9c908dff86
# ╟─5f817d98-e8a3-4599-a205-0b3c7b5fc3aa
# ╟─def52718-7c23-4516-8e2f-7e5d66aadfa5
# ╟─236b584a-65c7-435b-8c17-c87bb27a5837
# ╟─cecfaaba-da03-48a9-a0a2-1cd1dde8291b
# ╟─77b3f8bc-4861-4030-872e-ee1005fc9297
# ╟─5d126701-ac46-4beb-883c-2d32770ec46a
# ╟─90e4b9e2-47f1-48da-a205-69e170467add
# ╟─3e33aef8-3acb-4a46-9c0a-3e010ccca1ad
# ╟─103c2689-a09d-4e23-9d24-7c17caac74ae
# ╟─4e2c7812-c899-423a-bfa0-e05722923d82
# ╟─ca8fd5f7-a456-4c4f-9fcd-9c186808bd13
# ╟─95da7f31-b5ad-4c02-adcf-07408ac77e0b
# ╟─9b3755d0-3ce3-451b-ba8b-aacda52b6ed0
# ╟─6d0f023e-7be5-42d9-b8fc-f3f790b549c3
# ╟─a3fc98d0-2e29-43d4-842f-b87bf33087f7
# ╟─d77c5bb6-0cde-4a77-a829-b14d4f6121c7
# ╟─3200be1e-bae9-4381-a022-8d6eab313548
# ╟─1900afc5-47fa-45cb-9652-18d636c9f9de
# ╟─548f3fb1-6159-4cf2-878c-6a7b4f719122
# ╠═b44a8ae0-9279-432e-8b14-e4665146c710
# ╠═387bb56d-c382-4a81-9bed-9780ad0406df
# ╠═a1a8ec12-fd2f-4427-b0af-cc8c640cb848
# ╟─9e5ffbe3-9f89-44b4-a62e-d1ab83b36c95
# ╟─229e3112-2701-418f-be81-675cf4ada803
# ╟─3780fdb0-b4fb-44bf-bd70-665bfeac3a33
# ╠═5b6cdfc5-3db7-4c98-ab71-2d38552f0cb7
# ╠═f2583a51-7c2f-4462-99b6-a1832508f721
# ╠═6df575c4-e5e5-4ca0-9df9-b6c3048ecd8c
# ╠═13ea4cc6-ffcf-4b9a-a236-e2ec50f293bd
# ╠═b21f2230-b2f5-4cd7-b001-f833fe9ce8e2
# ╟─97652feb-4b21-4665-8dca-571d476d2320
# ╟─99dd0219-069e-498a-abdf-b359ae868ea3
# ╟─e5fde0eb-c15d-400b-a0d0-4e5157fd618e
# ╠═1623c8b9-cb84-4839-8e4e-fbbd7644c215
# ╟─cfc0dc88-f8ba-4c44-901f-cb3d3a6bfe27
# ╟─f272e9e7-0084-4ac8-a675-feae3b34d6bd
# ╟─63e87785-64d5-4dec-be38-b8d20d22b4ca
# ╟─93b0d805-afb8-49b1-898e-e400d1dee96c
# ╠═2021c063-a562-4333-8f4e-a55c18eb3dec
# ╠═ffb60616-6ef3-4af0-a61a-c412d23b2092
# ╠═8d2f6103-4035-415a-8000-37f53201bd9f
# ╟─4e48b49e-1963-4c53-aa16-bc124b4b8235
# ╟─43f6ddd5-8e4d-4dc8-8fb7-983033c240bc
# ╟─b6a45689-4bdf-4263-a229-0b7e38e5cac2
# ╟─660c4ab9-14ac-4604-b60f-b999c027a2b7
# ╟─29176cf7-dcd6-48c3-bfb6-056ea44ed1f5
# ╠═cc2993ca-7b78-406a-bcd0-068bc9f8f200
# ╟─b02abf85-4a48-4643-aea6-cee18e1fb0d4
# ╟─1a20824c-db94-4116-8bc0-d2fb9a736821
# ╟─d8c3f5fb-907d-47d3-9be6-b536f9326599
# ╟─4f783416-3603-401d-aed4-f8d53501691a
# ╟─04ffb19b-eefb-4c11-a664-ebb89fd1670f
# ╟─3e5e5803-6e9a-4860-a59c-bdfb56fd4973
# ╟─993685c3-7244-454c-91da-f4e013920fe7
# ╟─ba2fd3d2-28da-4cb2-85fa-cf7630d1aef6
# ╟─bed0d657-b10f-427e-93f8-0c1cf71c6a76
# ╟─201993e6-baac-40f0-9f07-f55e19cad371
# ╟─603bb8de-b7e1-4bf7-9001-40017508eece
# ╟─f2925f54-7382-4ba7-85ee-f57bc9de57d1
# ╟─0b95ac76-9075-4854-a78d-24ea07806dfa
# ╟─11cc06a2-f93b-4271-ad29-cbd1124925af
# ╟─01ea84dc-7f24-4c45-b4ef-de3b4283af87
# ╟─d0b4aef3-0d4b-4370-a059-af28a31b1955
# ╟─67a34131-abbf-44a6-9a83-0a88ed9dbf47
# ╟─0b035086-013c-4884-82b4-59a8425cc0af
# ╟─a1a353df-a89b-466e-b0c7-b07db96c2756
# ╟─06203d38-d9a7-489e-a655-186187e6c9a0
# ╟─bc5be9f6-ee45-4641-b26b-295ce4f0aca7
# ╟─302ae7c1-b9e3-4a11-9241-d5db87e392a6
# ╟─380e21e7-17f8-41f6-ac68-da8be570888c
# ╟─53a4ec18-2f8f-4d35-b01c-b07488088856
# ╟─3a356aa0-b4d0-4400-a610-47510a8a4d0b
# ╟─4db45dfa-045b-4b59-ace7-6ddee0f375f4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
