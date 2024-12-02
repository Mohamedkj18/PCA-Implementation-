import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import numpy as np


def plot_vector_as_image(image, h, w):
	"""
	function to plot a vector as image.
	Args:
	image - vector of pixels
	h, w - dimesnions of original pi
	"""
	plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
	plt.title('title', size=12)
	plt.show()

def get_pictures_by_name(name = None):
	"""
	Given a name returns all the pictures of the person with this specific name.
	YOU CAN CHANGE THIS FUNCTION!
	THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
	"""
	lfw_people = load_data()
	selected_images = []
	n_samples, h, w = lfw_people.images.shape
	target_label = list(lfw_people.target_names).index(name) if name else 0
	for image, target in zip(lfw_people.images, lfw_people.target):
		if (target == target_label):
			image_vector = image.reshape((h*w, 1))
			selected_images.append(image_vector)
	return selected_images, h, w

def load_data():
	lfw_people = fetch_lfw_people(min_faces_per_person=51, resize=0.4)
	return lfw_people



def PCA(X, k):
	"""
	Compute PCA on the given matrix.

	Args:
		X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
		For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
		k - number of eigenvectors to return

	Returns:
	  U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors
	  		of the covariance matrix.
	  S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
	"""
	n = X.shape[0]
	sigma = np.dot(X.T, X)/n

	U , S, V = np.linalg.svd(sigma)

	return V[:k], S[:k]


def experiment_1():
	"""In this experiment we reduce the matrix dimention to 10, and plot the pictures"""
	person , h, w = get_pictures_by_name()
	twoD_list = [[item[0] for item in picture] for picture in person ]
	matrix =np.array(twoD_list)
	V , _ = PCA(matrix, 10)
	for ev in V:	
		plot_vector_as_image(ev, h, w)

def experiment_2():
	"""In this experiment we reduce the matrix dimention to each k in {1,5,10,15,30,50,100}, plot the pictures 
	and compute the sum of the norms and plot it as a function of k"""
	person , h, w = get_pictures_by_name()
	twoD_list = [[item[0] for item in picture] for picture in person ]
	matrix =np.array(twoD_list)
	K= [1, 5,10,15,30,50,100]
	norm_sums = [0, 0 ,0, 0, 0, 0, 0]
	pictures = [person[i] for i in range(5)]
	V , _ = PCA(matrix, 100)
	for i , k in enumerate(K):
		U = V[:k]
		for x in pictures:
			u = np.dot(U.T, np.dot(U, x))
			norm_sums[i] += np.linalg.norm(x-u)
			plot_vector_as_image(x , h ,w)
			plot_vector_as_image(u, h ,w)
	plt.plot(K, norm_sums)
	plt.xlabel("k values")
	plt.ylabel("ℓ2 distances")
	plt.title("ℓ2 distances between the original vector and the reduced vector")
	plt.show()
	

if __name__ == "__main__":
	experiment_1()
	experiment_2()


