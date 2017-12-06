library(pheatmap)
library(ggplot2)
library(RColorBrewer)
library(gplots)
library(flashClust)
library(pracma)
nor01 <- function(x) {(x-min(x,na.rm=TRUE))/(max(x,na.rm=TRUE) - min(x,na.rm=TRUE))}
#load data
setwd("/Users/taylorsmith/Google Drive/Documents/Work/Workspace/Kidney Pathology/KDIGO_eGFR_traj/RESULTS/icu/subset1_results")
data <- read.table("ex_kdigo_dm_square.csv",header = TRUE, row.names = 1,sep=',')
data.norm <- data
data.m <- as.matrix(data.norm)

sq = squareform(data.m)
#dist and clustering
dd <- as.dist(data.norm)
my.dist.map <- dd
hc <- hclust(dd, method = 'ward')
#cut tree at height and output the right order

#reorder input matrix
order <- hc$order
hc_data <- data.norm[order,order]
hc_data.m <- as.matrix(hc_data)
#output the order
write.table(hc_data.m,"dm_matorder.txt", sep="\t")

colors <- c(seq(0,0.2,length=100),seq(0.2,0.6,length=100))
my_palette <- colorRampPalette(c("yellow","red","green"))(n=299)

#dengrogram of hclust
pdf('dist_hist.pdf')
hist(data.m,breaks=50)
dev.off()

pdf('dend.pdf')
plot(hc,cex=0.2)
dev.off()

#need to prompt user for number of clusters, k
clusters <- cutree(hc,k =6)
write.table(clusters[hc$order],"clusters_matorder.csv", sep=",")
write.table(clusters,"clusters_inorder.csv", sep=",")

#need to update this to be dynamic

cluster1=data[as.logical(clusters==1),as.logical(clusters==1)]
cluster2=data[as.logical(clusters==2),as.logical(clusters==2)]
cluster3=data[as.logical(clusters==3),as.logical(clusters==3)]
cluster4=data[as.logical(clusters==4),as.logical(clusters==4)]
cluster5=data[as.logical(clusters==5),as.logical(clusters==5)]
cluster6=data[as.logical(clusters==6),as.logical(clusters==6)]

write.table(cluster1,"cluster1_dm.csv", sep=",")
write.table(cluster2,"cluster2_dm.csv", sep=",")
write.table(cluster3,"cluster3_dm.csv", sep=",")
write.table(cluster4,"cluster4_dm.csv", sep=",")
write.table(cluster5,"cluster5_dm.csv", sep=",")
write.table(cluster6,"cluster6_dm.csv", sep=",")

cluster1 <- as.matrix(cluster1)
cluster2 <- as.matrix(cluster2)
cluster3 <- as.matrix(cluster3)
cluster4 <- as.matrix(cluster4)
cluster5 <- as.matrix(cluster5)
cluster6 <- as.matrix(cluster6)

which.min(colSums(cluster1))
which.min(colSums(cluster2))
which.min(colSums(cluster3))
which.min(colSums(cluster4))
which.min(colSums(cluster5))
which.min(colSums(cluster6))

#input
#pdf('hmap_orig.pdf')
#heatmap.2( data.m, Rowv=FALSE, Colv=FALSE, dendrogram='none',  notecol="black", col=my_palette,
#           trace='none', key=FALSE,lwid = c(.01,.99),lhei = c(.01,.99),margins = c(5,15), symkey=FALSE, symm=FALSE, symbreaks=TRUE, scale="none")
#dev.off()

#output
pdf('dist_hmap.pdf')
heatmap.2(hc_data.m, Rowv=FALSE, Colv=FALSE, dendrogram='none', notecol="black", col=my_palette,
          trace='none', key=FALSE,lwid = c(.01,.99),lhei = c(.01,.99),margins = c(5,15), symkey=FALSE, symm=FALSE, symbreaks=TRUE, scale="none")
dev.off()
#load svd results
#svd <- read.table("svd_resultnoaki6.txt", header = TRUE, row.names = 1)
#qplot(svd$svd_column1, svd$svd_column2,  colour = factor(svd$kidney), size = factor(svd$kidney),xlab="SVD1",ylab="SVD2")
#qplot(svd$svd_column1, svd$svd_column2, data = svd$kidney, colour = colors)
#distant matrix rearrage by clusters we have