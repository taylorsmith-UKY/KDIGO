#!/usr/bin/env Rscript

tpath = file.path("temp")
sink(file.path(tpath, "trash.txt"))
library(PredictABEL)
library(pROC)
# library(epiR)


labels = as.numeric(unlist(read.table(file.path(tpath, "labels.txt"))))
probs1 = as.numeric(unlist(read.table(file.path(tpath, "model1.txt"))))
probs2 = as.numeric(unlist(read.table(file.path(tpath, "model2.txt"))))
d = data.frame(a=labels,b=labels,c=labels,d=labels)

roc_obj1 <- roc(labels, probs1)
roc_obj2 <- roc(labels, probs2)
unlink(file.path(tpath, "trash.txt"))

sink(file.path(tpath, "reclass.txt"))
reclassification(d, 1, probs1, probs2, c(0, 0.05, 0.1, 0.2, 1.0))
improveProb(probs1, probs2, labels)
auc(roc_obj1)
ci.auc(roc_obj1)
auc(roc_obj2)
ci.auc(roc_obj2)
roc.test(roc_obj1, roc_obj2)
# epi.tests(probs1)
# epi.tests(probs2)
# sink()
# unlink(file.path(tpath, "reclass.txt"))
