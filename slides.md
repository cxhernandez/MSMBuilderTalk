% title: Conformational Dynamics in Mixtape
% author: Kyle A. Beauchamp

---
title: Analyzing Molecular Dynamics Data, Circa 1980

- Traditionally, MD Analysis happened in walled gardens (Gromacs, Amber, VMD)
- Duplication of statistical algorithms by non-experts (e.g. computational chemists)
- Possible code maintainability issues?


---
title: Jarvis Patrick Clustering in Gromacs

<pre class="prettyprint" data-lang="c">

static void jarvis_patrick(int n1, real **mat, int M, int P,
real rmsdcut, t_clusters *clust)
{
t_dist *row;
t_clustid *c;
int **nnb;
int i, j, k, cid, diff, max;
gmx_bool bChange;
real **mcpy = NULL;
if (rmsdcut < 0)
{
rmsdcut = 10000;
}
/* First we sort the entries in the RMSD matrix row by row.
* This gives us the nearest neighbor list.
*/

</pre>


---
title: Jarvis Patrick Clustering in Gromacs (Cont.)

<pre class="prettyprint" data-lang="c">

snew(nnb, n1);
snew(row, n1);
for (i = 0; (i < n1); i++)
{
for (j = 0; (j < n1); j++)
{
row[j].j = j;
row[j].dist = mat[i][j];
}
qsort(row, n1, sizeof(row[0]), rms_dist_comp);
if (M > 0)
{
/* Put the M nearest neighbors in the list */
snew(nnb[i], M+1);
for (j = k = 0; (k < M) && (j < n1) && (mat[i][row[j].j] < rmsdcut); j++)
{



</pre>



---
title: Jarvis Patrick Clustering in Gromacs (Cont.)

<pre class="prettyprint" data-lang="c">

// Five more pages of this
// You get the idea
// Also, how the hell do we even use this function?!
static void jarvis_patrick(int n1, real **mat, int M, int P, real rmsdcut, t_clusters *clust)


</pre>



---
title: Enter Data Science

- Thousands of experts are using machine learning approaches
- Well-tested, performant, and facile implementations are available
- Writing your own XYZ is bad science!



---
title: Enter Data Science

<pre class="prettyprint" data-lang="python">

import sklearn.cluster

clusterer = sklearn.cluster.KMeans(n_clusters=8)
cluster.fit(X)

</pre>

---
title: Mixtape: Philosophy

- Build on sklearn idioms
- Model(), fit(), transform(), Pipeline()


---
title: Everything is a model!


<pre class="prettyprint" data-lang="python">

import mixtape.cluster
clusterer = mixtape.cluster.KMeans(n_clusters=8)

import mixtape.tica
tica = mixtape.tica.tICA()

import mixtape.markovstatemodel
msm = mixtape.markovstatemodel.MarkovStateModel()

</pre>


---
title: Models need fitting!

<pre class="prettyprint" data-lang="python">

import mixtape.markovstatemodel
msm = mixtape.markovstatemodel.MarkovStateModel()
trajectories = [np.array([0, 0, 0, 1, 1, 1, 0 , 0])]
msm.fit(trajectories)
msm.transmat_

</pre>

Calculation outputs *always* have trailing underscores!


---
title: Models need fitting!

<pre class="prettyprint" data-lang="python">

import mixtape.cluster
clusterer = mixtape.cluster.KMeans(n_clusters=8, n_init=10)
trajectories = [np.random.normal(size=(100, 3))]
clusterer.fit(trajectories)
clusterer.cluster_centers_

</pre>

Input hyperparameters in the constructor!


---
title: Models can transform() data!

<pre class="prettyprint" data-lang="python">

import mixtape.cluster
clusterer = mixtape.cluster.KMeans(n_clusters=8, n_init=10)
trajectories = [np.random.normal(size=(100, 3))]
clusterer.fit(trajectories)
clusterer.transform([np.zeros(1, 3)])

[array([7], dtype=int32)]

</pre>


---
title: Pipelining Models


<pre class="prettyprint" data-lang="python">

import mixtape.
clusterer = mixtape.cluster.KMeans(n_clusters=8, n_init=10)
trajectories = [np.random.normal(size=(100, 3))]
clusterer.fit(trajectories)
clusterer.transform([np.zeros(1, 3)])

[array([7], dtype=int32)]

</pre>
