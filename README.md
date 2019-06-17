# Traffic Management - AI FOR S.E.A. by Grab
## Author: Karsten Eckhardt

### How to use it
This script exposes a relatively simple API. The first step is to clone the repository and install all requirements.

```
git clone https://github.com/R4h4/AIforSEA_Traffic_Management.git
cd AIforSEA_Traffic_Management
pip install -r requirements.txt
```

Next step is to produce the actual prediction:
```
python main.py -f <filename> -d <t-day> -t <t-time>
```
   - _Filename_, string, needs to be the full name with ending, and if not saved in the same directory, it also needs to include the full path.
   - _t-day_, int, the day of T
   - _t-time_, string, the timestamp of T, in the format "%H:%M"
   
The output is a CSV file in the base folder called "predictions.csv", containing the geohashes as columns and the 5 prediction timesteps as rows.

### Process
You can find the initial exploratory steps incl findings and description in the notebook [1_Exploratory_data_analysis](https://github.com/R4h4/AIforSEA_Traffic_Management/blob/master/notebooks/1_Exploratory_data_analysis.ipynb). 

Constructing the model: Generally, the problem at hand classifies as a time series forecasting problem. The challenge arises because we have multiple geolocations with different demand, which are highly likely to be related. The relationship manifests from the demand flow of one location to another, i.e., high demand at the fringes of a high-demand cluster like a city can lead to a high demand in areas within the city a few time steps later. However, the relationship might not be as strong as in regular traffic prediction, since demand doesn't automatically move from start to destination.

### Limitations
Long story short - there are lots and lots of them, mostly due to the time constraint
(I only learned about this challenge by the end of last week), decisions made at 4 am, and the fact that this was my first deep learning project in about 8 months. 
Here are some of them:
- Using geohashes for spatial partitioning is inferior to variable-sized
 Polygon tessellation (Davis, Raina, Jagannathan, 2018)[1]. This is an issue of the dataset itself.
- Gaps in the data: During the exploratory data analysis, it became clear that there are gaps in the location data, in some cases, next to higher demand clusters. Since we aggregate demand over the closest geohashes, this could lead to a perceived lower area demand for all areas around those white spots.
- Problematic encoding: Both hour and weekday where encoded as continuous integers during training, while they are technically categorical variables (hour 23 is not "worth" more than hour 6). The solution would be either to directly one-hot-encode those variables for the model input or even separate them, run them through an embedding layer and construct the model from the outputs of those two embedding layers as well as the remaining
data point.
- Possible suboptimal architecture: There are already some interesting papers on traffic forecasting out there, that can give a direction for better architectures. For example, Both Wang, Chen at al (2018)[2], and Zhao, Song et al. (2015)[3] describe graph based neural architectures for a similar problem. In particular, Graph Recurrent Neural Networks seem to outperform more traditional convolutional- and recurrent-networks for this particular problem.
- Suboptimal training method: As described above, this model has been trained with a new randomly generated dataset for every second epoch. Even if every random 
- No optimization: Mostly, due to time constraints, I did not manage to hyperparameter-tune the model. Since the first throw is seldom the best, it is, therefore, save to assume that even when staying in the current structure (encoder-decoder architecture), there are better performing parameters.

### Personal last words
This project has been exhausting but fun. For about 3 full days, I am not really satisfied with the result, and it pains me to release something that little tested, but it gave me a chance to get back into the machine- and in particular deep-learning world.


### Source
   - [1] https://arxiv.org/pdf/1812.03699.pdf
   - [2]
   - [3] https://arxiv.org/pdf/1811.05320.pdf