import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as scd
import matplotlib.pyplot as plt

np.set_printoptions(precision=5,suppress=True)
X=np.array([[2,3],[4,2],[5,1],[2,4],[6,3]])
#print(X.shape)
#plt.scatter(X[:,0],X[:,1])
#plt.show()
Z=sch.linkage(X,'single')
c, coph_dists=sch.cophenet(Z,scd.pdist(X))
print(c,coph_dists)
print(Z[0],Z[1],Z[2],Z[3],Z[-1:,2])
plt.figure(figsize=(25,10))
plt.title('hirarchical clustering')
plt.xlabel('sample index')
plt.ylabel('distance')
sch.dendrogram(
	Z,
	leaf_rotation=0.,
	leaf_font_size=12
)
plt.show()

