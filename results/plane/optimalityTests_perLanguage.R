library(lme4)
library(tidyr)
library(dplyr)
library(ggplot2)
depl = read.csv("../../grammars/dependency_length/total_summary_funchead_coarse.tsv", sep="\t")# %>% rename(Quality=AverageLength)
library(tidyr)
library(dplyr)
depl = depl %>% filter(grepl("FuncHead", ModelName)) %>% filter(grepl("Coarse", ModelName))
dataS = read.csv("../../grammars/plane/plane-fixed.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS2 = read.csv("../../grammars/plane/plane-fixed-best.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS3 = read.csv("../../grammars/plane/plane-fixed-best-large.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS4 = read.csv("../../grammars/plane/plane-fixed-random2.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS5 = read.csv("../../grammars/plane/plane-fixed-random3.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS6 = read.csv("../../grammars/plane/plane-fixed-random4.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS7 = read.csv("../../grammars/plane/plane-fixed-random5.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS = rbind(dataS, dataS2, dataS3, dataS4, dataS5, dataS6, dataS7)
dataP = read.csv("../../grammars/plane/plane-parse-redone.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS = dataS %>% group_by(Language, Type, Model) %>% summarise(Surprisal = mean(Surp, na.rm=TRUE))
dataP = dataP %>% group_by(Language, Type, Model) %>% summarise(Pars = mean(ParsU, na.rm=TRUE))
dataS = as.data.frame(dataS)
dataP = as.data.frame(dataP)
library(lme4)
dataS = dataS %>% mutate(Type = as.character(Type))
dataP = dataP %>% mutate(Type = as.character(Type))
dataS = dataS %>% mutate(Model = as.character(Model))
dataP = dataP %>% mutate(Model = as.character(Model))
dataS = dataS %>% mutate(Language = as.character(Language))
dataP = dataP %>% mutate(Language = as.character(Language))
data = merge(dataS, dataP, by=c("Language", "Model", "Type"), all.x=TRUE, all.y=TRUE)


data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM2", "manual_output_funchead_RANDOM", as.character(Type)))
data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM3", "manual_output_funchead_RANDOM", as.character(Type)))
data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM4", "manual_output_funchead_RANDOM", as.character(Type)))
data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM5", "manual_output_funchead_RANDOM", as.character(Type)))


dataBaseline = data %>% filter(Type == "manual_output_funchead_RANDOM")
dataGround = data %>% filter(Type == "manual_output_funchead_ground_coarse_final") %>% select(Language, Surprisal, Pars) %>% rename(SurprisalGround = Surprisal) %>% rename(ParsGround = Pars) %>% mutate(EffGround = ParsGround + 0.9*SurprisalGround) %>% group_by(Language)
data = merge(dataBaseline, dataGround, by=c("Language"))

data$Eff = data$Pars + 0.9*data$Surprisal

u = data %>% group_by(Language) %>% summarise(BetterSurprisal = sum(Surprisal > SurprisalGround, na.rm=TRUE), WorseSurprisal = sum(Surprisal <= SurprisalGround, na.rm=TRUE), TotalSurprisal = BetterSurprisal+WorseSurprisal, BetterFracSurprisal = BetterSurprisal/TotalSurprisal, BetterPars = sum(Pars > ParsGround, na.rm=TRUE), WorsePars = sum(Pars <= ParsGround, na.rm=TRUE), TotalPars = BetterPars+WorsePars, BetterFracPars = BetterPars/TotalPars, BetterEff = sum(Eff > EffGround, na.rm=TRUE), WorseEff = sum(Eff <= EffGround, na.rm=TRUE), TotalEff = BetterEff+WorseEff, BetterFracEff = BetterEff/TotalEff )


pValuesPars_bin = c()
pValuesPars_t = c()

pValuesSurprisal_bin = c()
pValuesSurprisal_t = c()

pValuesEff_bin = c()
pValuesEff_t = c()

sink("per-language-statistics.tex")
for(i in (1:51)) {
        testSurprisal_bin = binom.test(u$BetterSurprisal[[i]], u$TotalSurprisal[[i]], alternative="greater")
	pValuesSurprisal_bin = c(pValuesSurprisal_bin, testSurprisal_bin$p.value)
	language = u$Language[[i]]
        v = data %>% filter(Language == language)
        testSurprisal_t = t.test(v$Surprisal, mu=mean(v$SurprisalGround), alternative="greater")
	pValuesSurprisal_t = c(pValuesSurprisal_t, testSurprisal_t$p.value)
	if(u$TotalEff[[i]] == 0) {
   	   pValuesEff_bin = c(pValuesEff_bin, 0.5)
	} else {
  	   pValuesEff_bin = c(pValuesEff_bin, binom.test(u$BetterEff[[i]], u$TotalEff[[i]], alternative="greater")$p.value)
	}
	language = u$Language[[i]]
        v = data %>% filter(Language == language)
        if(is.na(mean(v$EffGround))) {
        	pValuesEff_t = c(pValuesEff_t, 0.5)
        } else {
        	pValuesEff_t = c(pValuesEff_t, t.test(v$Eff, mu=mean(v$EffGround), alternative="greater")$p.value)
        }
	if(u$TotalPars[[i]] == 0) {
               testPars_bin = binom.test(0,1, alternative="greater")
	       pValuesPars_bin = c(pValuesPars_bin, 0.5)
	} else {
                testPars_bin = binom.test(u$BetterPars[[i]], u$TotalPars[[i]], alternative="greater")
         	pValuesPars_bin = c(pValuesPars_bin, testPars_bin$p.value)
	}
	language = u$Language[[i]]
        v = data %>% filter(Language == language)
        if(is.na(mean(v$ParsGround))) {
                testPars_t = t.test(c(-1,1), mu=0, alternative="greater")
        	pValuesPars_t = c(pValuesPars_t, 0.5)
        } else {
                testPars_t = t.test(v$Pars, mu=mean(v$ParsGround), alternative="greater")
        	pValuesPars_t = c(pValuesPars_t, testPars_t$p.value)
        }
        cat(gsub("_", " ", language ), " & ")
        cat(paste("\\num{", format.pval(testSurprisal_t$p.value, digits=3), "}", sep=""))
        cat(" & ")
        cat(paste("\\num{", format.pval(testPars_t$p.value, digits=3), "}", sep=""))
        cat(" & ")
        cat(paste("$", round(testSurprisal_bin$estimate,2), "$ & $", paste("[", round(testSurprisal_bin$conf.int[[1]],2), ",", round(testSurprisal_bin$conf.int[[2]],2), "]", "$", sep=""), sep=""))
        cat(paste(" & ", "\\num{", format.pval(testSurprisal_bin$p.value, digits=3), "}", sep=""))
        cat(" & ")
        cat(paste("$", round(testPars_bin$estimate,2), "$ & $", paste("[", round(testPars_bin$conf.int[[1]],2), ",", round(testPars_bin$conf.int[[2]],2), "]", "$", sep=""), sep=""))
        cat(" & ")
        cat("\\num{", format.pval(testPars_bin$p.value, digits=3),"}", sep="")

#        cat(paste(round(testSurprisal_t$estimate,2), paste("[", round(testSurprisal_t$conf.int[[1]],2), ",", round(testSurprisal_t$conf.int[[2]],2), "]", sep=""), format.pval(testSurprisal_t$p.value, digits=3), sep=" & "))
#        cat(paste(round(testPars_t$estimate,2), paste("[", round(testPars_t$conf.int[[1]],2), ",", round(testPars_t$conf.int[[2]],2), "]", sep=""), format.pval(testPars_t$p.value, digits=3), sep=" & "))

        cat("\\\\ \n")
}
sink()
sink()
sink()
u$pValuesPars_bin = pValuesPars_bin
u$pValuesPars_t = pValuesPars_t

u$pValuesEff_bin = pValuesEff_bin
u$pValuesEff_t = pValuesEff_t

u$pValuesSurprisal_bin = pValuesSurprisal_bin
u$pValuesSurprisal_t = pValuesSurprisal_t


mean(u$pValuesPars_t < 0.05)
mean(u$pValuesSurprisal_t < 0.05)
mean(u$pValuesPars_t < 0.025 | u$pValuesSurprisal_t < 0.025)
mean(u$pValuesEff_t < 0.05)

# Hochberg's step-up procedure
parse = sort(u$pValuesPars_t)
limit = 0.05/(51-(1:51)+1)
mean(parse <= limit)
# TODO Czech missing data for parseability!? doesn't come out at 100%

surp = sort(u$pValuesSurprisal_t)
limit = 0.05/(51-(1:51)+1)
mean(surp <= limit)

either = sort(pmin(u$pValuesSurprisal_t, u$pValuesPars_t))*2
limit = 0.05/(51-(1:51)+1)
mean(either <= limit)


mean(u$pValuesPars_bin < 0.05)
mean(u$pValuesSurprisal_bin < 0.05)
mean(u$pValuesPars_bin < 0.025 | u$pValuesSurprisal_bin < 0.025)
mean(u$pValuesEff_bin < 0.05)

# Hochberg's step-up procedure
parse = sort(u$pValuesPars_bin)
limit = 0.05/(51-(1:51)+1)
mean(parse <= limit)

surp = sort(u$pValuesSurprisal_bin)
limit = 0.05/(51-(1:51)+1)
mean(surp <= limit)

either = sort(pmin(u$pValuesSurprisal_bin, u$pValuesPars_bin))*2
limit = 0.05/(51-(1:51)+1)
mean(either <= limit)


 








