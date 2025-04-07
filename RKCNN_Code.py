# Define RKCNN and KCNN
class RandomKConditionalNeighbors:
    def __init__(self, n_neighbors=5, h=10, r=5, m=0.5, smoothing=1, **kwargs):
        self.n_neighbors = n_neighbors
        self.h = h
        self.r = r
        self.m = m
        self.smoothing = smoothing
        self.models = []
        self.feature_subsets = []
        self.separation_scores = []
        self.weights = []
        self.label_encoder = LabelEncoder()

    def calculate_their_separation_score(self, X, y, feature_subset):
        """
        Calculates the separation score using the method described in the paper.
        """
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        centers = [np.mean(X[y == i], axis=0) for i in set(y)]
        overall_mean = np.mean(X, axis=0)

        A = np.array([(center - overall_mean) ** 2 for center in centers])

        B = []
        Nc = []
        for i in set(y):
            class_data = X[y == i]
            Nc.extend([1 / class_data.shape[0]] * class_data.shape[0])
            for _, row in class_data.iterrows():
                B.append((row - centers[list(set(y)).index(i)]) ** 2)
        B = np.array(B)
        Nc = np.array(Nc)

        a = np.array([1 if feature in feature_subset else 0 for feature in X.columns])

        aA = np.sqrt(np.dot(A, a)).sum()
        aB = np.sqrt(np.dot(B, a) * Nc).sum()

        return aA / aB if aB != 0 else 0

    def fit(self, X, y):
        """
        Fit the RKCNN model by generating and training KCNN models on feature subsets.
        """
        if self.r > self.h:
            raise ValueError(f"Invalid parameters: r ({self.r}) cannot be larger than h ({self.h}).")

        self.models = []
        self.feature_subsets = []
        self.separation_scores = []

        # Ensure y is a pandas Series for compatibility
        y = pd.Series(y) if not isinstance(y, pd.Series) else y

        # Fit the LabelEncoder with the target labels
        self.label_encoder.fit(y)

        for _ in range(self.h):
            num_features = int(len(X.columns) * self.m) if isinstance(self.m, float) else self.m;
            subset = X.sample(n=num_features, axis=1)
            self.feature_subsets.append(subset.columns)

            score = self.calculate_their_separation_score(subset, y, subset.columns)
            self.separation_scores.append(score)

        subset_scores = pd.DataFrame({
            "subset": self.feature_subsets,
            "score": self.separation_scores,
        }).sort_values(by="score", ascending=False).iloc[:self.r]

        self.feature_subsets = subset_scores["subset"].tolist()
        self.separation_scores = subset_scores["score"].tolist()

        for subset in self.feature_subsets:
            subset_X = X.loc[:, subset]
            model = KConditionalNeighbors(n_neighbors=self.n_neighbors, smoothing=self.smoothing)
            model.fit(subset_X, y)
            self.models.append(model)

        total_score = sum(self.separation_scores)
        self.weights = [score / total_score for score in self.separation_scores]

    def predict_proba(self, X):
        """
        Predict class probabilities by aggregating predictions from subset-specific models.
        """
        final_probabilities = pd.DataFrame(0, index=X.index, columns=self.label_encoder.classes_)

        for model, weight, subset in zip(self.models, self.weights, self.feature_subsets):
            subset_X = X.loc[:, subset]
            subset_probabilities = pd.DataFrame(
                model.predict_proba(subset_X),
                index=X.index,
                columns=self.label_encoder.classes_,
            )
            final_probabilities += subset_probabilities * weight

        return final_probabilities.div(final_probabilities.sum(axis=1), axis=0)

    def predict(self, X):
        """
        Predict class labels based on aggregated probabilities.
        """
        probabilities = self.predict_proba(X)
        return probabilities.idxmax(axis=1)

class KConditionalNeighbors(KNeighborsClassifier):
    def __init__(self, n_neighbors=5, smoothing=1, metric='minkowski', **kwargs):
        super().__init__(n_neighbors=n_neighbors, metric=metric, **kwargs)
        self.smoothing = smoothing
        self.label_encoder = LabelEncoder()

    def fit(self, X, y):
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        super().fit(X.to_numpy(), y_encoded)
        self._y = np.array(y)
        self._y_encoded = y_encoded
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        probabilities = np.zeros((X.shape[0], len(self.classes_)))
        skipped_classes = set()

        for class_index, class_label in enumerate(self.classes_):
            encoded_class_label = self.label_encoder.transform([class_label])[0]
            class_knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=self.metric)
            class_data = self._fit_X[self._y_encoded == encoded_class_label]

            if class_data.shape[0] == 0:
                skipped_classes.add(class_label)
                continue

            class_knn.fit(class_data, self._y_encoded[self._y_encoded == encoded_class_label])
            class_distances, _ = class_knn.kneighbors(X.to_numpy())

            class_prob_k = np.zeros((X.shape[0], self.n_neighbors))
            for k_index in range(self.n_neighbors):
                kth_distances = class_distances[:, k_index]
                denom = np.sum(self.n_neighbors / (kth_distances + self.smoothing) ** (self.p / self.n_features_in_))
                if denom > 0:
                    class_prob_k[:, k_index] = (self.n_neighbors / (kth_distances + self.smoothing) ** (self.p / self.n_features_in_)) / denom
                else:
                    class_prob_k[:, k_index] = 1 / self.n_neighbors

            probabilities[:, class_index] = np.mean(class_prob_k, axis=1)

        if skipped_classes:
            print(f"WARNING: The following classes were skipped due to lack of samples: {sorted(skipped_classes)}")

        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        probabilities = probabilities / row_sums
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        predictions = self.classes_[np.argmax(probabilities, axis=1)]
        return predictions
