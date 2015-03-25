package org.conan.mymahout.recommendation.job;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.DataModelBuilder;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.GenericBooleanPrefDataModel;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericBooleanPrefItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericBooleanPrefUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.impl.similarity.CityBlockSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.SpearmanCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.UncenteredCosineSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

/**
 * 
 * @author bsspirit@gmail.com
 * @link http://blog.fens.me/mahout-recommendation-api/
 */
public final class RecommendFactory {

	/**
	 * build Data model from file
	 */
	public static DataModel buildDataModel(String file) throws TasteException,
			IOException {
		return new FileDataModel(new File(file));
	}

	public static DataModel buildDataModelNoPref(String file)
			throws TasteException, IOException {
		return new GenericBooleanPrefDataModel(
				GenericBooleanPrefDataModel.toDataMap(new FileDataModel(
						new File(file))));
	}

	public static DataModelBuilder buildDataModelNoPrefBuilder() {
		return new DataModelBuilder() {
			public DataModel buildDataModel(
					FastByIDMap<PreferenceArray> trainingData) {
				return new GenericBooleanPrefDataModel(
						GenericBooleanPrefDataModel.toDataMap(trainingData));
			}
		};
	}

	/**
	 * similarity
	 */
	public enum SIMILARITY {
		/*
		 * 基于皮尔逊相关系数计算相似度
		 */
		PEARSON,
		/*
		 * 基于欧几里德距离计算相似度
		 */
		EUCLIDEAN,
		/*
		 * 计算 Cosine 相似度，余弦距离 计算方法
		 */
		COSINE,
		/*
		 * 基于 Tanimoto 系数算法
		 */
		TANIMOTO,
		/*
		 * (对数似然相似度）相似度
		 */
		LOGLIKELIHOOD,
		/*
		 * 斯皮尔曼相关系数
		 */
		SPEARMAN,
		/*
		 * 基于曼哈顿距离 相似度
		 */
		CITYBLOCK,
		/*
		 * 0.8 mahout 中有，0.9 已经不用了 计算两个聚类中所有项距离中的最大距离
		 * 聚类的相似度用于两个不同的聚类之间的距离（类似坐标系内的距离）
		 * 
		 * 目前聚类之间的距离计算只包含以下两个实现（暂时没有更好的实现算法）
		 */
		FARTHEST_NEIGHBOR_CLUSTER,
		/*
		 * 0.8 mahout 中有，0.9 已经不用了 计算两个聚类中所有项距离中的最小距离
		 */
		NEAREST_NEIGHBOR_CLUSTER
	}

	/**
	 * @Title: userSimilarity
	 * @data:2015年3月25日上午9:59:53
	 * @author:zhanghongliang@hiveview.com
	 *
	 * @param type
	 *            基于用户相似度类型
	 * @param m
	 *            数据源
	 * @return
	 * @throws TasteException
	 */
	public static UserSimilarity userSimilarity(SIMILARITY type, DataModel m)
			throws TasteException {
		switch (type) {
		case PEARSON:
			return new PearsonCorrelationSimilarity(m);
		case COSINE:
			return new UncenteredCosineSimilarity(m);
		case TANIMOTO:
			return new TanimotoCoefficientSimilarity(m);
		case LOGLIKELIHOOD:
			return new LogLikelihoodSimilarity(m);
		case SPEARMAN:
			return new SpearmanCorrelationSimilarity(m);
		case CITYBLOCK:
			return new CityBlockSimilarity(m);
		case EUCLIDEAN:
		default:
			return new EuclideanDistanceSimilarity(m);
		}
	}

	/**
	 * @Title: itemSimilarity
	 * @data:2015年3月25日上午9:59:23
	 * @author:zhanghongliang@hiveview.com
	 *
	 * @param type
	 *            物品相似度类型
	 * @param m
	 *            数据源
	 * @return
	 * @throws TasteException
	 */
	public static ItemSimilarity itemSimilarity(SIMILARITY type, DataModel m)
			throws TasteException {
		switch (type) {
		case PEARSON:
			return new PearsonCorrelationSimilarity(m);
		case COSINE:
			return new UncenteredCosineSimilarity(m);
		case TANIMOTO:
			return new TanimotoCoefficientSimilarity(m);
		case LOGLIKELIHOOD:
			return new LogLikelihoodSimilarity(m);
		case CITYBLOCK:
			return new CityBlockSimilarity(m);
		case EUCLIDEAN:
		default:
			return new EuclideanDistanceSimilarity(m);
		}
	}

	// public static ClusterSimilarity clusterSimilarity(SIMILARITY type,
	// UserSimilarity us) throws TasteException {
	// switch (type) {
	// case NEAREST_NEIGHBOR_CLUSTER:
	// return new NearestNeighborClusterSimilarity(us);
	// case FARTHEST_NEIGHBOR_CLUSTER:
	// default:
	// return new FarthestNeighborClusterSimilarity(us);
	// }
	// }

	/**
	 * neighborhood
	 */
	public enum NEIGHBORHOOD {
		NEAREST, THRESHOLD
	}

	/**
	 * @Title: userNeighborhood
	 * @data:2015年3月25日上午9:58:06
	 * @author:zhanghongliang@hiveview.com
	 *
	 * @param type
	 *            用户邻居类型
	 * @param s
	 *            基于用户相似算法
	 * @param m
	 *            原始数据
	 * @param num
	 *            最近邻居个数
	 * @return
	 * @throws TasteException
	 */
	public static UserNeighborhood userNeighborhood(NEIGHBORHOOD type,
			UserSimilarity s, DataModel m, double num) throws TasteException {
		switch (type) {
		case NEAREST:
			return new NearestNUserNeighborhood((int) num, s, m);
		case THRESHOLD:
		default:
			return new ThresholdUserNeighborhood(num, s, m);
		}
	}

	/**
	 * recommendation
	 */
	public enum RECOMMENDER {
		USER, ITEM
	}

	/**
	 * @Title: userRecommender
	 * @data:2015年3月25日上午9:57:03
	 * @author:zhanghongliang@hiveview.com
	 *
	 * @param us
	 *            基于用户相似度算法
	 * @param un
	 *            最近邻居 算法
	 * @param pref
	 *            是否需要偏好值
	 * @return
	 * @throws TasteException
	 */
	public static RecommenderBuilder userRecommender(final UserSimilarity us,
			final UserNeighborhood un, boolean pref) throws TasteException {
		return pref ? new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)
					throws TasteException {
				return new GenericUserBasedRecommender(model, un, us);
			}
		} : new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)
					throws TasteException {
				return new GenericBooleanPrefUserBasedRecommender(model, un, us);
			}
		};
	}

	/**
	 * 
	 * @Title: itemRecommender
	 * @data:2015年3月25日上午9:56:09
	 * @author:zhanghongliang@hiveview.com
	 *
	 * @param is
	 *            基于物品相似度算法类
	 * @param pref
	 *            是否需要偏好值
	 * @return
	 * @throws TasteException
	 */
	public static RecommenderBuilder itemRecommender(final ItemSimilarity is,
			boolean pref) throws TasteException {
		return pref ? new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)
					throws TasteException {
				return new GenericItemBasedRecommender(model, is);
			}
		} : new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)
					throws TasteException {
				return new GenericBooleanPrefItemBasedRecommender(model, is);
			}
		};
	}

	// public static RecommenderBuilder slopeOneRecommender() throws
	// TasteException {
	// return new RecommenderBuilder() {
	// public Recommender buildRecommender(DataModel dataModel) throws
	// TasteException {
	// return new SlopeOneRecommender(dataModel);
	// }
	//
	// };
	// }

	// public static RecommenderBuilder itemKNNRecommender(final ItemSimilarity
	// is, final Optimizer op, final int n) throws TasteException {
	// return new RecommenderBuilder() {
	// public Recommender buildRecommender(DataModel dataModel) throws
	// TasteException {
	// return new KnnItemBasedRecommender(dataModel, is, op, n);
	// }
	// };
	// }

	public static RecommenderBuilder svdRecommender(final Factorizer factorizer)
			throws TasteException {
		return new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel dataModel)
					throws TasteException {
				return new SVDRecommender(dataModel, factorizer);
			}
		};
	}

	// public static RecommenderBuilder treeClusterRecommender(final
	// ClusterSimilarity cs, final int n) throws TasteException {
	// return new RecommenderBuilder() {
	// public Recommender buildRecommender(DataModel dataModel) throws
	// TasteException {
	// return new TreeClusteringRecommender(dataModel, cs, n);
	// }
	// };
	// }

	public static void showItems(long uid,
			List<RecommendedItem> recommendations, boolean skip) {
		if (!skip || recommendations.size() > 0) {
			System.out.printf("uid:%s,", uid);
			for (RecommendedItem recommendation : recommendations) {
				System.out.printf("(%s,%f)", recommendation.getItemID(),
						recommendation.getValue());
			}
			System.out.println();
		}
	}

	/**
	 * evaluator
	 */
	public enum EVALUATOR {
		AVERAGE_ABSOLUTE_DIFFERENCE, RMS
	}

	public static RecommenderEvaluator buildEvaluator(EVALUATOR type) {
		switch (type) {
		case RMS:
			return new RMSRecommenderEvaluator();
		case AVERAGE_ABSOLUTE_DIFFERENCE:
		default:
			return new AverageAbsoluteDifferenceRecommenderEvaluator();
		}
	}

	public static void evaluate(EVALUATOR type, RecommenderBuilder rb,
			DataModelBuilder mb, DataModel dm, double trainPt)
			throws TasteException {
		System.out.printf("%s Evaluater Score:%s\n", type.toString(),
				buildEvaluator(type).evaluate(rb, mb, dm, trainPt, 1.0));
	}

	public static void evaluate(RecommenderEvaluator re, RecommenderBuilder rb,
			DataModelBuilder mb, DataModel dm, double trainPt)
			throws TasteException {
		System.out.printf("Evaluater Score:%s\n",
				re.evaluate(rb, mb, dm, trainPt, 1.0));
	}

	/**
	 * statsEvaluator
	 */
	public static void statsEvaluator(RecommenderBuilder rb,
			DataModelBuilder mb, DataModel m, int topn) throws TasteException {
		RecommenderIRStatsEvaluator evaluator = new GenericRecommenderIRStatsEvaluator();
		IRStatistics stats = evaluator.evaluate(rb, mb, m, null, topn,
				GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0);
		// System.out.printf("Recommender IR Evaluator: %s\n", stats);
		System.out.printf(
				"Recommender IR Evaluator: [Precision:%s,Recall:%s]\n",
				stats.getPrecision(), stats.getRecall());
	}

}
