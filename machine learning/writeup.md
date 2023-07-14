# Assignment 4 - Machine Learning!



## Part 3 - Written Questions



1.  **Question**: Explain your K-Means algorithm. What are the parts of your algorithm, and how do they work together to divide your samples into different clusters?

	**Answer**: *Your answer here*
My implementation of the Kmeans algorithm uses the procedure with starts by guessing (randomly selecting) the initial centroids, then the guess is refined repeatedly assigning examples to their closest centroids and then recomputing the centroids. This process was done in init_centroids, closest_centroids, and compute_centroids. Euclidean distance was used to measure the similarities between data points to help in finding the closest centroids. Lastly, the inertia function that I implemented measured how well the data set was clustered.
------------------------------  

2.

- **Question**: What is each data point that is being divided into clusters? what does each cluster represent?

	 **Answer**: *Your answer here*
Each data point that is being divided into clusters is the colors of the tiger image. Each cluster refers to a collection of data points aggregated together, and in this case that is a color (similar colors will be clustered together)


- **Question**: How does changing the number of clusters impact the result of the song dataset and the image compression dataset?

	**Answer**: *Your answer here*

------------------------------
As you increase the number of clusters, the inertia will go down. Inertia measures how well the dataset was clustered, and a lower inertia means the distance between each data point and its centroid are small. Increasing the number of clusters is only beneficial to a certain extent, once you reach a certain point (6 for song clusters) the inertia goes back up.

3.

- **Question**: What is the difference between supervised classification, supervised regression, and unsupervised learning?

	**Answer**: *Your answer here*
Supervised classification is a procedure for identifying similar areas on an image by that are representative of specific objects or classes. This can be done by identifying training sites and using those as reference of the classification of other pixels in an image.
Supervised regression is a learning technique which helps find the correlation between variables and enables us to predict the continuous output variable based on one or more predictor variables
Unsupervised learning is a technique in which the users do not need to "supervise" the model and instead the model works on its own to discover patterns and information that was previously undetected.

- **Question**: Give an example of an algorithm under each, specifying the type of data that the algorithm takes in and the goal of the algorithm, and an explanation for why they are a supervised classification/supervised regression/unsupervised algorithm.

	**Answer**: *Your answer here*
Supervised classification:
K Nearest Neighbor can be an algorithm example for supervised classification as a point can be classified based on which group is fits best with.
An example of supervised classification in the real world could be an algorithm that is trained to identify different types of birds. It would take in am image and hopefully be able to recognize the bird based on different patters/colors. This is supervised classification because it is learned from labeled data and trained to classify images.

Supervised regression:
K nearest neighbor could also be a supervised regression algorithm as feature similarity can lead to certain predictions. KNN takes in new data and hopes to fit it with existing groups.
An algorithm that is trained to predict stock market patterns. The algorithm could take in a specific business you want to predict trends for and the goal would be to model a credible trajectory. This is supervised regression because the data taken in is labeled and the algorithm is trying to find correlation between how the business does in relation to different events (going public, consumer spending habits, etc.)

Unsupervised algorithm:
Kmeans is an example of unsupervised learning, as the algorithm takes in data and hopes to find similarities through clustering. This is unsupervised because it is working on its own to discover patterns without labels.
------------------------------

4. **Question**: Give an overview of how you would modify your Kmeans class to implement Fair K-Means in  `kmeans.py`. Describe any methods you would add and where you would call them. You don’t need to understand the mathematical details of the Fair K-Means algorithm to have a general idea of the key changes necessary to modify your solution.

	**Answer**: *Your answer here*
The Fair-Kmeans technique considers cluster similarities to the centroid as well as considering which cluster will be skewed the least in terms of sensitive attributes. To try to implement this into my kmeans implementation I would try to find a way to account for sensitive attributes and also create a method that would measure how these attributes could be skewed and call it in compute_centroids, so that before a centroid is assigned to a cluster it will have been checked whether or not it will affect the fair representation in clustering.
------------------------------

5. **Question**:  How does the Fair K-means algorithm define fairness? Describe a situation or context where this definition of fairness might not be appropriate, or match your own perception of fairness.

	**Answer**: *Your answer here*
The Fair K-means algorithm defines fairness by trying to make sure that all sensitive groups are accounted for. They define their method to be fairer on the basis that the clusters created will more proportionally respect the demographic characteristics of the overall dataset. In the case of weeding out resumes this may promote fairness, but this could have the opposite affect when a sensitive demographic is overly represented in a negative way. For example, when examining crime and who is committing crime, making sure that racial demographics are accounted for in clusters could be harmful as this could create a skewed perception (and one that already exists) that one racial group is more dangerous than another.
------------------------------

6. **Question**: Are there any situations in which even a perfectly fair ML system might still cause or even exacerbate harm? Are there other metrics or areas of social impact you might consider? Justify your opinion.

	**Answer**: *Your answer here*
Our society today is plagued with inequality. When creating any ML system I think it is important to examine the findings and consider potential reasons for why a pattern has emerged. A perfectly fair ML system could take in data points from developing countries and see that men have higher test scores and income rate than women, which is true statistically, but there are social aspects which have caused this result. The lack of educational opportunities for women and low expectations for their ability to succeed contribute to their lower performances which cannot be seen or made fair by a ML system. I think that it is always important to follow quantitative data with qualitative data.
------------------------------

7. **Question**:
	Based on the text, “Algorithms, if left unchecked, can create highly skewed and homogenous clusters that do not represent the demographics of the dataset. ”

	a. Identify a situation where an algorithm has caused significant social repercussions (you can talk about established companies or even any algorithms that you have encountered in your daily life).

	b. Based on your secondary knowledge of the situation what happened, why do you think it happened, and how would you do things differently (if you would). Justify your choices.

	**Answer**: *Your answer here*
a. I think that there are many examples of biased algorithms. One that comes to mind is the use of facial recognition technology and how the algorithms for much of this technology were trained with not enough equal representational imagery, so people of color often were recognized with less accuracy. This reflects the social inequality within our society. Facial recognition technology is also prevalent in the use of police surveillance and bias in that technology could lead to people of certain minority groups being targeted because the technology was trained under a certain bias.

b. I think the problem with the facial recognition algorithm happened because of the lack of representation of all demographics in the training data. I think it is important to train algorithms with a wide variety of diverse data so that all members of the population are represented and represented in a equal manner.

------------------------------


8. **Question**:
	Read the article and answer the following questions:

	a. How did Google respond to this?

	b. Experiment with the autocomplete feature of Google now. What do you notice now? How do you feel about the way Google handled this solution? List one area in which Google can still improve.

	**Answer**: *Your answer here*

  a. In response, google said that they would take down any autocomplete predictions are hurtful against groups and has asked users to report issues especially when it is "hateful".

  b. Experimenting with the autocomplete feature, I typed in a variety of things pertaining to race, gender, and politics and all the predictions seemed to be very objective and not based on societal stereotypes. I think that google handled the situation well, but should do more to combat these things from popping up rather than just asking the user base to report it when they see something hurtful. In the article we read it mentioned how there needs to be more fair representation online and information should not be monopolized by google. While google will unlikely relinquish its power, it should try to empower minority groups and groups that need more representation. While objectivity is good in some ways, going about in a "colorblind" manner will not help fix the inequalities we face. 
