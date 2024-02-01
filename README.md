# Fake News Detection
 Detecting fake news with TFIDF and Passive Aggressive Classifier.
 
 Dataset: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data

## Training without any process.
First we are reading csv's then we are adding labels to each dataframes. Because it doesn't have labels in it.
```
#Reading the fake news csv.
fake_df = pd.read_csv("dataset/Fake.csv")

#Adding 'FAKE' label.
fake_df['label'] = 'FAKE'

#Reading the true news csv.
true_df = pd.read_csv("dataset/True.csv")

#Adding 'TRUE' label.
true_df['label'] = 'TRUE'
```

Then we are getting together dataframes and shuffling rows with the help of sample function. I set every random state to 26 because I want to get accurate results. 
And why 26? Because I just like the number 26 ¯\ _(ツ)_/¯
```
#Merging these dataframes and shuffling rows with 'sample' function.
merged_df = pd.concat([fake_df, true_df])
merged_df = merged_df.sample(frac=1, random_state=26)
```

After we splitting dataset, initializing and training TFIDF Vectorizer and Passive Aggressive Classifier.
```
#Splitting dataset.
x_train, x_test, y_train, y_test = train_test_split(merged_df['text'], merged_df['label'], test_size=0.2, random_state=26)
#Initialize tfidf vectorizer.
tfidf_vec = TfidfVectorizer(stop_words='english', max_df=0.7)

#Fit-transform the train data and just transform the test data.
tfidf_train = tfidf_vec.fit_transform(x_train)
tfidf_test = tfidf_vec.transform(x_test)

#Initialize passive aggressive classifier.
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)
```

Until this project I didn't know anything about Passive Aggressive algorithm. Simply it is an online algorithm that suitable for large stream of data. E.g. if you are getting real time data from Twitter/X or another social media platform this algorithm works for it.

It called Passive Aggressive because if algorithm predicts correctly it does nothing, stays passive. But if made an incorrect predict then it moves weight vector aggressively to just make the decision perfect. If you want to know much about this algorithm I prefer this video: https://www.youtube.com/watch?v=TJU8NfDdqNQ

After training we are ready to get results.
```
#Make prediction on the test set and calculate accuracy.
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#Build confusion matrix.
confusion_matrix(y_test, y_pred, labels=['FAKE', 'TRUE'])
```
It's accuracy 99.48% for me. Actually it is pretty high. But if you just take a look to the True datas you can see nearly every single of them starts with '*A location name* (Reuters) -'. And I think it causes biases and unreal results. So first I am going to delete (Reuters) substrings then checking results again. Then I erase every '*A location name* (Reuters) -' pattern in the data.

## Training with removing '(Reuters)' brand

With this simple loop we are erasing (Reuters) substrings in every text.
```
for i in range(len(true_df['text'])):
    true_df['text'][i] = true_df['text'][i].replace('(Reuters)', '')
```

After this we are following same steps.
```
#Merging dataframes and shuffling rows with 'sample' function.
merged_df = pd.concat([fake_df, true_df])
merged_df = merged_df.sample(frac=1, random_state=26)

#Splitting dataset.
x_train, x_test, y_train, y_test = train_test_split(merged_df['text'], merged_df['label'], test_size=0.2, random_state=26)

#Initialize tfidf vectorizer.
tfidf_vec2 = TfidfVectorizer(stop_words='english', max_df=0.7)

#Fit-transform the train data and just transform the test data.
tfidf_train = tfidf_vec2.fit_transform(x_train)
tfidf_test = tfidf_vec2.transform(x_test)

#Initialize passive aggressive classifier.
pac2 = PassiveAggressiveClassifier(max_iter=50)
pac2.fit(tfidf_train, y_train)

#Make prediction on the test set and calculate accuracy.
y_pred = pac2.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')
```
Now you can see our accuract is slightly decreased. Now it should be 98.89%. I want to see how our accuracy decrasing if we remove location names as well.

## Training with removing locations and (Reuters) brand

First we have to reread True.csv again to get original data.
```
#Reading the fake news csv.
true_df = pd.read_csv("dataset/True.csv")
#Adding 'TRUE' label.
true_df['label'] = 'TRUE'
```

With this loop we are splitting texts and removing locations and (Reuters) brand.
```
seperator = '(Reuters) -'
for i in range(len(true_df['text'])):
    if true_df['text'][i].find(seperator) == -1:
        continue
    true_df['text'][i] = true_df['text'][i].split(seperator, 1)[1]
```

We are repeating same steps again.
```
#Merging dataframes and shuffling rows with 'sample' function.
merged_df = pd.concat([fake_df, true_df])
merged_df = merged_df.sample(frac=1, random_state=26)

#Splitting dataset.
x_train, x_test, y_train, y_test = train_test_split(merged_df['text'], merged_df['label'], test_size=0.2, random_state=26)

#Initialize tfidf vectorizer.
tfidf_vec3 = TfidfVectorizer(stop_words='english', max_df=0.7)

#Fit-transform the train data and just transform the test data.
tfidf_train = tfidf_vec3.fit_transform(x_train)
tfidf_test = tfidf_vec3.transform(x_test)

#Initialize passive aggressive classifier.
pac3 = PassiveAggressiveClassifier(max_iter=50)
pac3.fit(tfidf_train, y_train)

#Make prediction on the test set and calculate accuracy.
y_pred = pac3.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')
```
Now we are getting 98.82% accuracy. It nearly didn't changed at all.

## Testing with different dataset

Finally I would like to test with completely different dataset.

I used this dataset: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

Now I need to inform you about a mistake in data card about this dataset. It says label 1 is refers to True, 0 is Fake. But it is wrong. You can see in discussion.

So we'll replace labels.
```
new_set = pd.read_csv("for_test/WELFake_Dataset.csv")

#Replacing labels 0 to TRUE, 1 to FAKE
new_set['label'] = new_set['label'].replace(0,'TRUE')
new_set['label'] = new_set['label'].replace(1,'FAKE')
```

This dataset includes some null values. We have to check that and remove them.
```
#To check if we have any null values.
new_set['text'].isna().sort_values()

new_set = new_set.dropna(axis=0, subset=['text'])
new_set['text'].isna().sort_values()
```

Now we are good to go. First I want to try the first trained model. Which is trained withouth any process.

```
#First try with the model that we trained first way
new_test = tfidf_vec.transform(new_set['text'])

y_pred = pac.predict(new_test)
score = accuracy_score(new_set['label'], y_pred)
print(f'Accuracy: {round(score*100,2)}%')
```
Accuracy score is 83.05% for me.

Ok. Let's try the model we trained with the last method.

```
#Now try with the last method
new_test = tfidf_vec3.transform(new_set['text'])

y_pred = pac3.predict(new_test)
score = accuracy_score(new_set['label'], y_pred)
print(f'Accuracy: {round(score*100,2)}%')
```

Accuracy is 83.72% this time. So the rearrangement that we did is worked. It predict better.
