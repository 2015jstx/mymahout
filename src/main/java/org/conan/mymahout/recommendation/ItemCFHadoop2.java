/*  
 * @(#) ItemCFHadoop2.java Create on 2015年3月24日 上午9:50:00   
 *   
 * Copyright 2015 .hiveview
 */


package org.conan.mymahout.recommendation;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.mahout.cf.taste.hadoop.item.RecommenderJob;

/**
 * @ItemCFHadoop2.java
 * @created at 2015年3月24日 上午9:50:00 by zhanghongliang@hiveview.com
 *
 * @desc
 *
 * @author  zhanghongliang@hiveview.com
 * @version $Revision$
 * @update: $Date$
 */
public class ItemCFHadoop2 {

    public static void main(String[] args) throws Exception {
    	String input = "/user/hdfs/userCF";
    	String output = "hdfs/userCF/test";
    	String tempDir = "hdfs/userCF/tmp";
		String similarityClassname = "org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.LoglikelihoodSimilarity";
		String booleanData = "false";
		args = new String[]{input,output,tempDir ,booleanData,similarityClassname};
        JobConf conf = new JobConf(ItemCFHadoop.class);
        GenericOptionsParser optionParser = new GenericOptionsParser(conf, args);
        String[] remainingArgs = optionParser.getRemainingArgs();
        if (remainingArgs.length != 5) {
            System.out.println("args length: "+remainingArgs.length);
            System.err.println("Usage: hadoop jar <jarname> <package>.ItemCFHadoop <inputpath> <outputpath> <tmppath> <booleanData> <similarityClassname>");
            System.exit(2);
        }
        
        System.out.println("input : "+remainingArgs[0]);
        System.out.println("output : "+remainingArgs[1]);
        System.out.println("tempdir : "+remainingArgs[2]);
        System.out.println("booleanData : "+remainingArgs[3]);
        System.out.println("similarityClassname : "+remainingArgs[4]);
        
        StringBuilder sb = new StringBuilder();
        sb.append("--input ").append(remainingArgs[0]);
        sb.append(" --output ").append(remainingArgs[1]);
        sb.append(" --tempDir ").append(remainingArgs[2]);
        sb.append(" --booleanData ").append(remainingArgs[3]);
        sb.append(" --similarityClassname ").append(remainingArgs[4]);
        conf.setJobName("ItemCFHadoop");
        RecommenderJob job = new RecommenderJob();
        job.setConf(conf);
        job.run(sb.toString().split(" "));
    }
}
