In this Assignment I used PCA on pictures of faces. 

As a necessary first step I was asked to implement PCA without the use of the Sklearn
PCA method (or any other package that implements PCA). 

Afterwards, I selected one random person with enough pictures (> 50) from the pool of people. I implemented  a
utility function that will help load pictures of a specific person.

Then I Constructed a matrix X whose rows are the flattened images of this person. 
Then preformerd two experiments:
1. Ran PCA on X with k = 10 and plotted each of the 10 eigenvectors as pictures.
2. Ran PCA For k = 1, 5, 10, 30, 50, 100 and reduceed the
dimension using PCA to dimension k. Selected at random 5 pictures (the same 5 pictures
for all values of k) and plotted the each of the 5 original pictures next to the pictures
obtained by transforming the reduced pictures back to the original dimension.
Also plotted, as a function of k, the sum (over the entire dataset) of the ℓ2 distances between
the two.
