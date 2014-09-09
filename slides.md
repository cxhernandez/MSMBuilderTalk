% title: Conformational Dynamics in Mixtape
% author: Kyle A. Beauchamp
% author: September 3, 2014
% thankyou: Thanks!
% thankyou_details: Contributors: Robert M., Kyle B., Bharath R., Matt H., Steve K., Gert K., Muneeb S.
% contact: <span>www</span> <a href="http://msmbuilder-mixtape.s3-website-us-west-1.amazonaws.com/latest/index.html">mixtape docs</a>
% contact: <span>github</span> <a href="https://github.com/rmcgibbo/mixtape">mixtape</a>

---
title: Old-School Analysis of MD Data

- Analysis happens in walled gardens (Gromacs, Amber, VMD)
- Exclusively command line interfaces, C and Fortran code
- Duplication of statistical algorithms by non-experts (e.g. chemists)
- Possible code maintainability issues?

---
title: Jarvis Patrick Clustering in Gromacs
subtitle: <a href="https://github.com/gromacs/gromacs/blob/master/src/gromacs/gmxana/gmx_cluster.c">real code in gromacs</a>

<pre class="prettyprint" data-lang="c">

static void jarvis_patrick(int n1, real **mat, int M, int P,
real rmsdcut, t_clusters *clust) {
t_dist *row;
t_clustid *c;
int **nnb;
int i, j, k, cid, diff, max;
gmx_bool bChange;
real **mcpy = NULL;
if (rmsdcut < 0) {
rmsdcut = 10000;
}
/* First we sort the entries in the RMSD matrix row by row.
* This gives us the nearest neighbor list.
*/
</pre>

---
title: Jarvis Patrick Clustering in Gromacs (Cont.)

<pre class="prettyprint" data-lang="c">

// Five more pages of this
// You get the idea

// Also, how do we even use this function?
static void jarvis_patrick(int n1, real **mat, int M, int P,
real rmsdcut, t_clusters *clust);
</pre>



---
title: Enter Data Science

- Machine learning is mainstream now!
- Thousands of experts are using machine learning approaches
- Well-tested, performant, and facile implementations are available
- Writing your own is not the way to go!
    * E.g: is clustering _that_ special and MD-specific such that
    we need our own custom algorithms and implementations? No. 


---
title: Mixtape: Philosophy

Let's build on [scikit-learn](http://scikit-learn.org/stable/) idioms:

 
- Everything is a ``Model()``!
- Models ``fit()`` data!
- Models ``transform()`` data!
- ``Pipeline()`` concatenates models!
- Encourage Best-Practices (cross-validation)

---
title: Everything is a <code>Model()</code>!


<pre class="prettyprint" data-lang="python">

>>> import mixtape.cluster
>>> clusterer = mixtape.cluster.KMeans(n_clusters=4)

>>> import mixtape.tica
>>> tica = mixtape.tica.tICA(n_components=3)

>>> import mixtape.markovstatemodel
>>> msm = mixtape.markovstatemodel.MarkovStateModel()

</pre>

Hyperparameters go in the constructor.


<footer class="source"> 
Actually, everything is a <code>sklearn.base.BaseEstimator()</code>
</footer>


---
title: Models <code>fit()</code> data!

<pre class="prettyprint" data-lang="python">

>>> import mixtape.cluster

>>> trajectories = [np.random.normal(size=(100, 3))]

>>> clusterer = mixtape.cluster.KMeans(n_clusters=4, n_init=10)
>>> clusterer.fit(trajectories)

>>> clusterer.cluster_centers_

array([[-0.22340896,  1.0745301 , -0.40222902],
       [-0.25410827, -0.11611431,  0.95394687],
       [ 1.34302485,  0.14004818,  0.01130485],
       [-0.59773874, -0.82508303, -0.95703567]])


</pre>

Estimated parameters *always* have trailing underscores!


---
title: <code>fit()</code> acts on lists of sequences


<pre class="prettyprint" data-lang="python">

>>> import mixtape.markovstatemodel

>>> trajectories = [np.array([0, 0, 0, 1, 1, 1, 0, 0])]

>>> msm = mixtape.markovstatemodel.MarkovStateModel()
>>> msm.fit(trajectories)

>>> msm.transmat_

array([[ 0.75      ,  0.25      ],
       [ 0.33333333,  0.66666667]])

</pre>

This is different from sklearn, which uses 2D arrays.


---
title: Models <code>transform()</code> data!

<pre class="prettyprint" data-lang="python">

>>> import mixtape.cluster

>>> trajectories = [np.random.normal(size=(100, 3))]

>>> clusterer = mixtape.cluster.KMeans(n_clusters=4, n_init=10)
>>> clusterer.fit(trajectories)
>>> Y = clusterer.transform(trajectories)

[array([5, 6, 6, 0, 5, 5, 1, 6, 1, 7, 5, 7, 4, 2, 2, 2, 5, 3, 0, 0, 1, 3, 0,
        5, 5, 0, 4, 0, 0, 3, 4, 7, 3, 5, 5, 5, 6, 1, 1, 0, 0, 7, 4, 4, 2, 6,
        1, 4, 2, 0, 2, 4, 4, 5, 2, 6, 3, 2, 0, 6, 3, 0, 7, 7, 7, 0, 0, 0, 3,
        3, 2, 7, 6, 7, 2, 5, 1, 0, 3, 6, 3, 2, 0, 5, 0, 3, 4, 2, 5, 4, 1, 5,
        5, 4, 3, 3, 7, 2, 1, 4], dtype=int32)]

</pre>
Moving the data-items from one "space" / representation into another.

---
title: <code>Pipeline()</code> concatenates models!


<pre class="prettyprint" data-lang="python">

>>> import mixtape.cluster, mixtape.markovstatemodel
>>> from sklearn.pipeline import Pipeline

>>> trajectories = [np.random.normal(size=(100, 3))]

>>> clusterer = mixtape.cluster.KMeans(n_clusters=2, n_init=10)
>>> msm = mixtape.markovstatemodel.MarkovStateModel()
>>> pipeline = Pipeline([("clusterer", clusterer), ("msm", msm)])

>>> pipeline.fit(trajectories)
>>> msm.transmat_

array([[ 0.53703704,  0.46296296],
       [ 0.53333333,  0.46666667]])

</pre>
Data "flows" through transformations in the pipeline.


---
title: Featurizing Trajectories

Featurizers wrap MDTraj functions via the `transform()` function

<div style="float:right;">
<img height=225 src=figures/rama.png />
</div>


<pre class="prettyprint" style="width:75%" data-lang="python">

>>> from mixtape.featurizer import DihedralFeaturizer
>>> from mixtape.datasets import fetch_alanine_dipeptide
>>> from matplotlib.pyplot import hexbin, plot

>>> trajectories = fetch_alanine_dipeptide()["trajectories"]
>>> featurizer = DihedralFeaturizer(
...        ["phi", "psi"], sincos=False)
>>> X = featurizer.transform(trajectories)
>>> phi, psi = np.rad2deg(np.concatenate(X).T)

>>> hexbin(phi, psi)
</pre>



---
title: Loading Trajectories

<pre class="prettyprint" data-lang="python">

>>> import glob
>>> import mdtraj as md

>>> filenames = glob.glob("./Trajectories/*.h5")
>>> trajectories = [md.load(filename) for filename in filenames]

</pre>

Note: for big datasets, you can get fancies with ``md.iterload``.

---
title: Old-school MSMs

<pre class="prettyprint" data-lang="python">

>>> import mdtraj as md
>>> from mixtape.featurizer import DihedralFeaturizer
>>> from mixtape.datasets import fetch_alanine_dipeptide
>>> from mixtape.cluster import KCenters
>>> from mixtape.markovstatemodel import MarkovStateModel
>>> from sklearn.pipeline import Pipeline

>>> trajectories = fetch_alanine_dipeptide()["trajectories"]

>>> cluster = KCenters(n_clusters=10, metric=md.rmsd)
>>> msm = MarkovStateModel()
>>> pipeline = Pipeline([("cluster", cluster), ("msm", msm)])

>>> pipeline.fit(trajectories)
</pre>

---
title: Old-school MSMs (contd.)

<pre class="prettyprint" data-lang="python">
# ...
>>> featurizer = DihedralFeaturizer(["phi", "psi"], sincos=False)
>>> X = featurizer.transform(trajectories)
>>> phi, psi = np.rad2deg(np.concatenate(X).T)

>>> hexbin(phi, psi)
>>> phi, psi = np.rad2deg(featurizer.transform([cluster.cluster_centers_])[0].T)
>>> plot(phi, psi, 'w*', markersize=25)
</pre>

<center>
<img height=250 src="figures/rama-cluster-centers.png">
</center>

---
title: Cross Validation

<pre class="prettyprint" data-lang="python">

from sklearn.cross_validation import KFold

cv = KFold(len(trajectories), n_folds=5)

for fold, (train_index, test_index) in enumerate(cv):
    train_data = [trajectories[i] for i in train_index]
    test_data = [trajectories[i] for i in test_index]
    model.fit(train_data)
    model.score(test_data)
</pre>

Also scikit-learn's <code>GridSearchCV</code> and <code>RandomizedSearchCV</code>.


---
title: Model Scoring with GMRQ
subtitle: We can improve a lot over KMeans.

<center>
<img height=355 src=figures/SETD2_kcenters.png />
<img height=355 src=figures/SETD2_tICA_KMeans.png />
</center>



