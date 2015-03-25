package org.conan.mymahout.recommendation;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

public class UserCF {
	
	//最近邻居个数
    final static int NEIGHBORHOOD_NUM = 2;
    //推荐个数
    final static int RECOMMENDER_NUM = 3;

    public static void main(String[] args) throws IOException, TasteException {
		//定义文件 内容格式  用户id、物品id、偏好值
        String file = "datafile/item.csv";
      //用fileDataModel 会监控该文件的目录，有新数据时，放到该目录下，调用refresh 会加载新的内容
        DataModel model = new FileDataModel(new File(file));
        
     // 定义 基于欧几里德距离计算相似度
//        	PearsonCorrelationSimilarity：基于皮尔逊相关系数计算相似度
//        	EuclideanDistanceSimilarity：基于欧几里德距离计算相似度
//        	TanimotoCoefficientSimilarity：基于 Tanimoto 系数计算相似度
        UserSimilarity user = new EuclideanDistanceSimilarity(model);
        
        //o	NearestNUserNeighborhood：对每个用户取固定数量 N 的最近邻居，这里用 2
        //o	ThresholdUserNeighborhood：对每个用户基于一定的限制，取落在相似度门限内的所有用户为邻居。
        NearestNUserNeighborhood neighbor = new NearestNUserNeighborhood(NEIGHBORHOOD_NUM, user, model);
      //实例GenericUserBasedRecommender，实现 基于用户偏好的 推荐策略。
        Recommender r = new GenericUserBasedRecommender(model, neighbor, user);
        
        //获取用户的所以id
        LongPrimitiveIterator iter = model.getUserIDs();

        while (iter.hasNext()) {
            long uid = iter.nextLong();
            //获取用户 uid 的推荐物品id
            List<RecommendedItem> list = r.recommend(uid, RECOMMENDER_NUM);
            System.out.printf("uid:%s", uid);
            for (RecommendedItem ritem : list) {
                System.out.printf("(%s,%f)", ritem.getItemID(), ritem.getValue());
            }
            System.out.println();
        }
    }
}
