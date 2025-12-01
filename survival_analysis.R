# r/survival_analysis.R

library(tidyverse)
library(survival)
library(survminer)

data_path <- "data/processed/spoilage_clean.csv"
df <- read_csv(data_path)

# Expecting columns: duration, event (1 = spoiled, 0 = censored)
surv_obj <- Surv(time = df$duration, event = df$event)

# Basic Kaplan–Meier curve
km_fit <- survfit(surv_obj ~ 1)
ggsurvplot(km_fit, conf.int = TRUE, risk.table = TRUE)

# Cox proportional hazards model (example with temperature and pH)
cox_fit <- coxph(surv_obj ~ temp + ph, data = df)
summary(cox_fit)
