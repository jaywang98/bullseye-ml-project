import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import IncrementalPCA, PCA, TruncatedSVD
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


warnings.simplefilter(action='ignore', category=FutureWarning)


class DataProcessor:
    # load data, regularization, resample
    def __init__(self, random_state=5813):
        self.random_state = random_state
        self.raw_train_df, self.raw_test_df = None, None
        self.train_X, self.test_X, self.train_y, self.test_y = None, None, None, None
        self.train_vX, self.val_X, self.train_vy, self.val_y = None, None, None, None

    def load_data(self):
        self.raw_train_df = pd.read_excel("dataset/training.xlsx", sheet_name=0)
        self.raw_test_df = pd.read_excel("dataset/test.xlsx", sheet_name=0)

    def preprocess(self):
        train_ratio = 0.8
        test_ratio = 0.1
        val_ratio = 0.1
        scaler = MinMaxScaler()
        train_df_scaled = scaler.fit_transform(self.raw_train_df)
        X = train_df_scaled[:, :-1]
        y = train_df_scaled[:, -1]
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(
            X, y, test_size=test_ratio, random_state=self.random_state
        )
        train_pos_ratio = np.sum(self.train_y) / len(self.train_y)
        test_pos_ratio = np.sum(self.test_y) / len(self.test_y)
        print('Train set size: {}, train_pos_ratio: {:.3f}, test set size: {}, test_pos_ratio: {:.3f}'.format(
            len(self.train_X), train_pos_ratio, len(self.test_X), test_pos_ratio))

    def resample(self):
        resampling_ratio = 1.0
        train_X_df = pd.DataFrame(self.train_X)
        train_y_df = pd.DataFrame(self.train_y, columns=['Y'])
        train_df = pd.concat([train_X_df, train_y_df], axis=1)

        # class_balance = True
        train_df_1 = train_df[train_df.Y == 1]
        train_df_0 = train_df[train_df.Y == 0]

        print('Before resampling, train_df has {} $1$ instances and {} $0$ instances, the ratio is: {:.4f}.'.format(
            len(train_df_1), len(train_df_0), len(train_df_1) / len(train_df_0)))

        train_df_1_upsampled = resample(train_df_1, replace=True,
                                        n_samples=int(len(train_df_0) * resampling_ratio),
                                        random_state=self.random_state)

        train_df = pd.concat([train_df_1_upsampled, train_df_0]).reset_index(drop=True)
        train_df_1 = train_df[train_df.Y == 1]
        train_df_0 = train_df[train_df.Y == 0]
        print('After resampling, train_df has {} $1$ instances and {} $0$ instances, the ratio is: {:.4f}.'.format(
            len(train_df_1), len(train_df_0), len(train_df_1) / len(train_df_0)))

        self.train_X = train_df.iloc[:, :-1]
        self.train_y = train_df.iloc[:, -1]

        train_pos_ratio = np.sum(self.train_y) / len(self.train_y)
        test_pos_ratio = np.sum(self.test_y) / len(self.test_y)
        print('Train set size: {}, train_pos_ratio: {:.3f}, test set size: {}, test_pos_ratio: {:.3f}'.format(
            len(self.train_X), train_pos_ratio, len(self.test_X), test_pos_ratio))

    def pca_data(self):
        # draw relationship between Explained Variance Ratio and Number of Principal Components
        pca = PCA()
        pca.fit(self.train_X)
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        plt.vlines(x=6, ymax=1, ymin=0.5, colors="r", linestyles="--")
        plt.hlines(y=0.95, xmax=12, xmin=0, colors="g", linestyles="--")
        plt.title('Principal Components Number & Explained Variance Ratio')
        plt.ylabel('Explained Variance Ratio')
        plt.xlabel('Number of Principal Components')
        plt.plot(explained_variance)
        plt.savefig('pic/Relationship_PCA.jpg', dpi=300)
        plt.show()

        # draw Correlation_Matrix_PCA
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, 9))
        tmp_train_X = self.train_X
        corr_mat = np.corrcoef(tmp_train_X.transpose())
        sns.heatmap(corr_mat, ax=ax0)
        ax0.set_title('Original Correlation Matrix')

        pca_final = IncrementalPCA(n_components=6)
        tmp_train_X = pca_final.fit_transform(self.train_X)
        corr_mat = np.corrcoef(tmp_train_X.transpose())
        sns.heatmap(corr_mat, ax=ax1)
        ax1.set_title('PCA Correlation Matrix')
        plt.savefig('pic/Correlation_Matrix_PCA.jpg', dpi=300)
        plt.show()

    def format_data(self):
        self.load_data()
        self.preprocess()
        self.resample()
        # self.pca_data()

    def get_data(self):
        return self.train_X, self.test_X, self.train_y, self.test_y

    def get_val_data(self):
        return self.train_vX, self.val_X, self.train_vy, self.val_y


class ModelEval:
    def __init__(self,):
        self.data_processor = None
        self.model = None
        self.model_name = None
        self.class_balance = True
        self.train_X, self.train_y, self.test_X, self.test_y = None, None, None, None
        self.train_vX, self.val_X, self.train_vy, self.val_y = None, None, None, None

    # set model and model name
    def set_model(self, model, tmp_suffix):
        self.model = model
        if self.model.__class__.__name__ == 'RandomForestClassifier':
            prefix = 'RandomForestClassifier'
        else:
            prefix = self.model.estimator.__class__.__name__
        if self.class_balance:
            self.model_name = prefix + '_balanced'
        else:
            self.model_name = prefix
        if tmp_suffix is not None:
            self.model_name = self.model_name + tmp_suffix

    # get the dataset
    def set_data_processor(self, dp):
        self.data_processor = dp
        self.train_X, self.test_X, self.train_y, self.test_y = dp.get_data()
        self.train_vX, self.val_X, self.train_vy, self.val_y = dp.get_data()

    def grid_search(self, model, grid_params):
        cv = StratifiedKFold(n_splits=10)
        clf = GridSearchCV(model, grid_params, cv=cv)
        clf.fit(self.train_X, self.train_y)
        param = list(grid_params.keys())[0]
        plt.plot(np.ma.getdata(clf.cv_results_['param_' + param]),
                 np.ma.getdata(clf.cv_results_['mean_test_score']))
        plt.xlabel(param)
        plt.ylabel('mean test score')
        plt.title(clf.estimator.__class__.__name__ + " Grid Search Curve")
        plt.savefig('pic/' + clf.estimator.__class__.__name__ + '_GridSearch_Curve.jpg', dpi=300)
        plt.show()
        return clf

    def __eval_model(self):
        self.model.fit(self.train_X, self.train_y)
        train_acc = self.model.score(self.train_X, self.train_y)
        test_acc = self.model.score(self.test_X, self.test_y)
        return train_acc, test_acc

    def __model_performance(self):
        y_pred = self.model.predict(self.test_X)
        acc = accuracy_score(y_pred, self.test_y)
        prec = precision_score(y_pred, self.test_y)
        recall = recall_score(y_pred, self.test_y)
        f1 = f1_score(y_pred, self.test_y)
        roc_auc = roc_auc_score(y_pred, self.test_y)
        return {'model': self.model_name,
                'accuracy': acc,
                'precision': prec,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc}

    def plot_roc_curve(self):
        if self.model.estimator.__class__.__name__ == 'KNeighborsClassifier':
            y_score = self.model.predict_proba(self.test_X)[:, 1]
        else:
            y_score = self.model.decision_function(self.test_X)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(self.test_y.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.figure()
        lw = 2
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            color="darkorange",
            lw=lw,
            label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(self.model.estimator.__class__.__name__ + " ROC curve")
        plt.legend(loc="lower right")
        plt.savefig('pic/' + self.model.estimator.__class__.__name__ + '_ROC_Curve.jpg', dpi=300)
        plt.show()

    # User only need to apply this function to eval model's performance
    def performance_measure(self):
        train_acc, test_acc = self.__eval_model()
        print(self.model_name)
        print('\t train_acc: {}, test_acc: {}'.format(train_acc, test_acc))
        performance_dict = self.__model_performance()
        return performance_dict


class ModelSet:
    def __init__(self):
        self.model_eval = None
        self.data_processor = None
        self.performance_df = columns = ['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    def set_model_eval(self, me):
        self.model_eval = me

    def set_data_processor(self, dp):
        self.data_processor = dp

    def test_logistic(self):
        logistic_model = LogisticRegression(max_iter=2000)
        param_grid = {'C': np.arange(0.1, 10., 0.05)}
        res_model = self.model_eval.grid_search(logistic_model, param_grid)
        modelname_suffix = '_' + str(res_model.best_params_)
        self.model_eval.set_model(res_model, modelname_suffix)
        performance_dict = self.model_eval.performance_measure()
        self.performance_df.append(performance_dict)
        self.model_eval.plot_roc_curve()
        return self.performance_df

    def test_ridge(self):
        ridge_model = RidgeClassifier(max_iter=2000)
        param_grid = {'alpha': np.arange(0.1, 10., 0.05)}
        res_model = self.model_eval.grid_search(ridge_model, param_grid)
        modelname_suffix = '_' + str(res_model.best_params_)
        self.model_eval.set_model(res_model, modelname_suffix)
        performance_dict = self.model_eval.performance_measure()
        self.performance_df.append(performance_dict)
        self.model_eval.plot_roc_curve()
        return self.performance_df

    def test_lda(self):
        lda_model = LinearDiscriminantAnalysis(solver='eigen')
        param_grid = {'shrinkage': np.arange(0.01, 1., 0.01)}
        res_model = self.model_eval.grid_search(lda_model, param_grid)
        modelname_suffix = '_' + str(res_model.best_params_)
        self.model_eval.set_model(res_model, modelname_suffix)
        performance_dict = self.model_eval.performance_measure()
        self.performance_df.append(performance_dict)
        self.model_eval.plot_roc_curve()
        return self.performance_df

    def test_qda(self):
        qda_model = QuadraticDiscriminantAnalysis(store_covariance=True)
        param_grid = {'reg_param': np.arange(0.01, 5., 0.01)}
        res_model = self.model_eval.grid_search(qda_model, param_grid)
        modelname_suffix = '_' + str(res_model.best_params_)
        self.model_eval.set_model(res_model, modelname_suffix)
        performance_dict = self.model_eval.performance_measure()
        self.performance_df.append(performance_dict)
        self.model_eval.plot_roc_curve()
        return self.performance_df

    def test_knn(self):
        knn_model = KNeighborsClassifier(algorithm='ball_tree', weights='distance', n_neighbors=18)
        param_grid = {'n_neighbors': np.arange(1, 20, 1)}
        res_model = self.model_eval.grid_search(knn_model, param_grid)
        modelname_suffix = '_' + str(res_model.best_params_)
        self.model_eval.set_model(res_model, modelname_suffix)
        performance_dict = self.model_eval.performance_measure()
        self.performance_df.append(performance_dict)
        self.model_eval.plot_roc_curve()
        return self.performance_df

    def test_random_forest(self):
        criterions = ['gini', 'entropy', 'log_loss']
        max_depths = 3 + np.arange(15)
        for criterion in criterions:
            for depth in max_depths:
                rf_model = RandomForestClassifier(criterion=criterion, max_depth=depth)
                model_suf = '_{}_({})'.format(criterion, depth)
                self.model_eval.set_model(rf_model, model_suf)
                performance_dict = self.model_eval.performance_measure()
                self.performance_df.append(performance_dict)
        return self.performance_df

    def draw_svc_decision_edge(self):
        # draw SVC decision edge
        X, X_test, Y, Y_test = self.data_processor.get_data()
        reducer = PCA(n_components=2)
        train_X_pca = reducer.fit_transform(X)
        C_2d_range = [1e-1, 1, 1e1]
        gamma_2d_range = [1, 20, 84]
        classifiers = []
        for C in C_2d_range:
            for gamma in gamma_2d_range:
                clf = SVC(C=C, gamma=gamma)
                clf.fit(train_X_pca, Y)
                classifiers.append((C, gamma, clf))
        plt.figure(figsize=(16, 12))
        xx, yy = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1.1, 200))
        for k, (C, gamma, clf) in enumerate(classifiers):
            # evaluate decision function in a grid
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # visualize decision function for these parameters
            plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
            plt.title("gamma=%d, C=%d" % (gamma, C), size="medium")

            # visualize parameter's effect on decision function
            plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
            plt.scatter(train_X_pca[:, 0], train_X_pca[:, 1], c=Y, cmap=plt.cm.RdBu_r, edgecolors="k")
            plt.xticks(())
            plt.yticks(())
            plt.axis("tight")
        plt.savefig('pic/SVC_Decision_Edge,jpg', dpi=300)
        plt.show()

    def test_svc(self):
        svc_model = SVC(kernel='rbf', C=1, max_iter=2000)
        param_grid = {
            'gamma': np.arange(0.1, 15, 1),
        }
        res_model = self.model_eval.grid_search(svc_model, param_grid)
        model_suf = '_{}_{}'.format('rbf', str(res_model.best_params_))
        self.model_eval.set_model(res_model, model_suf)
        performance_dict = self.model_eval.performance_measure()
        self.performance_df.append(performance_dict)
        self.model_eval.plot_roc_curve()
        self.draw_svc_decision_edge()
        return self.performance_df

    def plot_embedding_2d(self, in_x, y, title=None):
        """Plot an embedding X with the class label y colored by the domain d."""
        x_min, x_max = np.min(in_x, 0), np.max(in_x, 0)
        X = (in_x - x_min) / (x_max - x_min)

        # Plot colors numbers
        plt.figure(figsize=(9, 9))
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            # plot colored number
            plt.text(X[i, 0], X[i, 1], str(y[i]),
                     color=plt.cm.Set1(y[i]),
                     fontdict={"weight": 'bold', 'size': 9})

        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)
        plt.savefig('pic/kmeans.jpg', dpi=300)
        plt.show()

    def test_kmeans(self):
        # self.performance_df = columns = ['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        train_X, test_X, train_y, test_y = self.data_processor.get_data()
        inits = ['k-means++', 'random']
        n_components = np.arange(2, 13)
        low_ds = ["PCA", "SVD"]
        for init in inits:
            for low_d in low_ds:
                # PCA and SVD can reduce dim into any dimensions, so we put them together
                for n_component in n_components:
                    if low_d == "PCA":
                        reducer = PCA(n_components=n_component)
                        # train_X_pca = reducer.fit_transform(train_X)
                        # test_X_pca = reducer.fit_transform(test_X)
                        kmeans_model = KMeans(n_clusters=2, init=init, max_iter=1000)
                        y_pred = kmeans_model.fit_predict(test_X)
                        tmp_name = 'KMeans_[init: {}, PCA_n_comps: {}]'.format(init, n_component)
                        print(
                            'KMeans, init={}, PCA_n_comps:{}, test acc: {}, precision_score:{}, recall_score:{}, '
                            'f1_score:{}, roc_auc_score:{}'.format(
                                init, n_component, accuracy_score(y_pred, test_y), precision_score(y_pred, test_y),
                                recall_score(y_pred, test_y), f1_score(y_pred, test_y), roc_auc_score(y_pred, test_y)))
                    else:
                        reducer_1 = TruncatedSVD(random_state=0, n_components=n_component)
                        # train_X_svd = reducer_1.fit_transform(train_X)
                        # test_X_svd = reducer_1.fit_transform(test_X)
                        kmeans_model = KMeans(n_clusters=2, init=init, max_iter=1000)
                        y_pred = kmeans_model.fit_predict(test_X)
                        tmp_name = 'KMeans_[init: {}, SVD_n_comps: {}]'.format(init, n_component)
                        print(
                            'KMeans, init={}, SVD_n_comps:{}, test acc: {}, precision_score:{}, recall_score:{}, '
                            'f1_score:{}, roc_auc_score:{}'.format(
                                init, n_component, accuracy_score(y_pred, test_y), precision_score(y_pred, test_y),
                                recall_score(y_pred, test_y), f1_score(y_pred, test_y), roc_auc_score(y_pred, test_y)))
                    performance_dict = dict()
                    performance_dict['model'] = tmp_name
                    performance_dict['accuracy'] = accuracy_score(y_pred, test_y)
                    performance_dict['precision'] = precision_score(y_pred, test_y)
                    performance_dict['recall'] = recall_score(y_pred, test_y)
                    performance_dict['f1'] = f1_score(y_pred, test_y)
                    performance_dict['roc_auc'] = roc_auc_score(y_pred, test_y)
                    self.performance_df.append(performance_dict)

            for n_component in range(2, 4):
                reducer = TSNE(n_components=n_component, random_state=0)
                train_X_tsne = reducer.fit_transform(train_X)
                test_X_tsne = reducer.fit_transform(test_X)
                kmeans_model = KMeans(n_clusters=2, init=init, max_iter=1000)
                y_pred = kmeans_model.fit_predict(test_X)
                if n_component == 2:
                    tmp_name = "KMeans_[init:{}, TSNE_2_n_comps: {}]".format(init, n_component)
                    print(
                        'KMeans, init={}, TSNE_2_n_comps:{}, test acc: {}, precision_score:{}, recall_score:{}, '
                        'f1_score:{}, roc_auc_score:{}'.format(
                            init, n_component, accuracy_score(y_pred, test_y), precision_score(y_pred, test_y),
                            recall_score(y_pred, test_y), f1_score(y_pred, test_y), roc_auc_score(y_pred, test_y)))
                    self.plot_embedding_2d(test_X_tsne[:, 0:2], y_pred, "TSNE 2D")
                else:
                    continue
                performance_dict = dict()
                performance_dict['model'] = tmp_name
                performance_dict['accuracy'] = accuracy_score(y_pred, test_y)
                performance_dict['precision'] = precision_score(y_pred, test_y)
                performance_dict['recall'] = recall_score(y_pred, test_y)
                performance_dict['f1'] = f1_score(y_pred, test_y)
                performance_dict['roc_auc'] = roc_auc_score(y_pred, test_y)
                self.performance_df.append(performance_dict)
        return self.performance_df

    def test_neural_network(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=12))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        opt = Adam(learning_rate=1e-3)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics='accuracy')
        train_X, test_X, train_y, test_y = self.data_processor.get_data()
        hist = model.fit(train_X, train_y, epochs=100, batch_size=10, validation_split=0.1)
        plt.plot(hist.history['val_accuracy'], label='val_accuracy')
        plt.plot(hist.history['accuracy'], label='accuracy')
        plt.legend()
        plt.savefig('pic/NN.jpg', dpi=300)
        plt.show()

    def test_all_models(self):
        self.test_logistic()
        self.test_ridge()
        self.test_qda()
        self.test_knn()
        self.test_svc()
        self.test_random_forest()
        self.test_neural_network()

    def get_performance(self):
        return self.performance_df


if __name__ == "__main__":
    # Load Data
    dp = DataProcessor()
    dp.format_data()
    dp.pca_data()

    # Model Eval
    me = ModelEval()
    me.set_data_processor(dp)

    # Model Set
    ms = ModelSet()
    ms.set_model_eval(me)
    ms.set_data_processor(dp)

    # Get model performance
    ms.test_all_models()

    # Get and save models' performances
    df = ms.get_performance()[6:]
    df = pd.DataFrame(df)
    df.to_csv('dataset/model_performance.csv')
