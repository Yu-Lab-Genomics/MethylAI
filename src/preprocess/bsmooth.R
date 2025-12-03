########################################
arguments = commandArgs(T)
wd = arguments[1]
post_fix = arguments[2]
thread_number = as.integer(arguments[3])
output_sample_info_file = arguments[4]
output_methylation_file = arguments[5]
if(length(arguments) > 5) {
  smooth_ns = as.integer(arguments[6])
  smooth_h = as.integer(arguments[7])
} else {
  smooth_ns = 70
  smooth_h = 1000
}
########################################
library(bsseq)
library(BiocParallel)
library(data.table)
library(R.utils)
library(glue)
########################################
print(paste0("wd: ", wd))
print(paste0("post_fix: ", post_fix))
print(paste0("thread_number: ", thread_number))
print(paste0("output_sample_info_file: ", output_sample_info_file))
print(paste0("output_methylation_file: ", output_methylation_file))
print(paste0("smooth_ns: ", as.character(smooth_ns)))
print(paste0("smooth_h: ", as.character(smooth_h)))
########################################
read_methylation_bed = function(methylation_file, dataset_index) {
  print(glue("methylation_file: {methylation_file}, dataset_index: {dataset_index}"))
  methylation_bed = fread(
    methylation_file, header=T, sep="\t", quote="", nThread=thread_number
  )
  # generate BSseq object
  bs = BSseq(
    chr = methylation_bed$chr,
    pos = methylation_bed$start,
    M = as.matrix(methylation_bed$mc, ncol = 1),
    Cov = as.matrix(methylation_bed$cov, ncol = 1),
    sampleNames = dataset_index
  )
  return(bs)
}
########################################
#设置工作路径
setwd(wd)
#读取以.bed结尾的文件名
methylation_bed_file_vector = list.files(pattern=post_fix)
methylation_bed_file_dataframe = data.frame(
  dataset_index=1:length(methylation_bed_file_vector), methylation_file=methylation_bed_file_vector
)
#输出文件信息
write.table(methylation_bed_file_dataframe, file=output_sample_info_file,
            sep="\t", quote=F, row.names=F)
#读取第1个bed文件，作为combined_methylation_bs的初始化
combined_methylation_bs = read_methylation_bed(methylation_bed_file_vector[1], "1")
#使用for循环，combine所有bed文件
dataset_index = 2
for (methylation_dataset_index in dataset_index:length(methylation_bed_file_vector)) {
  methylation_file = methylation_bed_file_vector[methylation_dataset_index]
  new_methylation_bs = read_methylation_bed(methylation_file, as.character(methylation_dataset_index))
  combined_methylation_bs = combine(combined_methylation_bs, new_methylation_bs)
}
#平滑处理
smooth_combined_methylation_bs = BSmooth(
  combined_methylation_bs,
  ns = smooth_ns,
  h = smooth_h,
  BPPARAM = MulticoreParam(workers=thread_number, progressbar=TRUE)
)
#准备输出文件
#位置信息
output_position = data.frame(granges(smooth_combined_methylation_bs))[,c(1,2,3)]
output_position$end = output_position$end + 2
#output_dataframe：位置、平滑的甲基化、原始甲基化、测序深度
output_dataframe = data.frame(
  output_position,
  getMeth(smooth_combined_methylation_bs, type="smooth"),
  getMeth(smooth_combined_methylation_bs, type="raw"),
  getCoverage(smooth_combined_methylation_bs, type = "Cov")
)
output_dataframe$start = as.integer(output_dataframe$start)
output_dataframe$end = as.integer(output_dataframe$end)
#调整colnames
colnames(output_dataframe) = c(
  "chr", "start", "end",
  paste0("smooth_", methylation_bed_file_dataframe$dataset_index),
  paste0("raw_", methylation_bed_file_dataframe$dataset_index),
  paste0("coverage_", methylation_bed_file_dataframe$dataset_index)
)
#输出文件
if (! endsWith(output_methylation_file, ".gz")) {
  output_methylation_file = paste0(output_methylation_file, ".gz")
}
fwrite(output_dataframe, file=output_methylation_file, sep="\t", quote=F, row.names=F, na="NA")
print("Finish.")
