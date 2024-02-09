library(tidyverse)

idir <- sprintf("results_hpc")
ifile.list <- list.files(idir)
results <- do.call("rbind", lapply(ifile.list, function(ifile) {
    df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols(),
                     guess_max=2)
}))

results %>%
    group_by(n_train, model) %>%    
    summarise(test_mse_se=2*sd(test_mse)/sqrt(n()), test_mse=mean(test_mse)) %>%
    ggplot(aes(x=n_train, y=test_mse, color=model)) +
    geom_point() +
    geom_line() +
    geom_errorbar(aes(ymin=test_mse-test_mse_se, ymax=test_mse+test_mse_se)) +
    scale_x_continuous(trans='log10') +
    theme_bw()
