library(lme4)
library(tidyr)
library(dplyr)
library(ggplot2)
#depl = read.csv("../../../grammars/dependency_length/total_summary_funchead_coarse.tsv", sep="\t")# %>% rename(Quality=AverageLength)

# Read predictability estimates
dataS =  read.csv("../../../grammars/plane/controls/plane-fixed-pureUD.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS = dataS %>% group_by(Language, Type, Model) %>% summarise(Surp = mean(Surp, na.rm=TRUE))
dataS = as.data.frame(dataS)

# Read parseability estimates
dataP =  read.csv("../../../grammars/plane/controls/plane-parse-pureUD.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataP = dataP %>% group_by(Language, Type, Model) %>% summarise(UAS = mean(UAS, na.rm=TRUE), Pars = mean(Pars, na.rm=TRUE))
dataP = as.data.frame(dataP)

# Ensure everything is treated as a string, not a factor, to ensure the frames can be merged
dataS = dataS %>% mutate(Type = as.character(Type))
dataP = dataP %>% mutate(Type = as.character(Type))
dataS = dataS %>% mutate(Model = as.character(Model))
dataP = dataP %>% mutate(Model = as.character(Model))
dataS = dataS %>% mutate(Language = as.character(Language))
dataP = dataP %>% mutate(Language = as.character(Language))
data = merge(dataS, dataP, by=c("Language", "Model", "Type"), all.x=TRUE, all.y=TRUE)

# Collapse all five groups of random grammars (10 random grammars per group per language)
data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM_pureUD2", "manual_output_funchead_RANDOM_pureUD", as.character(Type)))
data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM_pureUD3", "manual_output_funchead_RANDOM_pureUD", as.character(Type)))
data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM_pureUD4", "manual_output_funchead_RANDOM_pureUD", as.character(Type)))
data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM_pureUD5", "manual_output_funchead_RANDOM_pureUD", as.character(Type)))


dataBaseline = data %>% filter(Type == "manual_output_funchead_RANDOM_pureUD")
dataGround = data %>% filter(Type == "manual_output_funchead_ground_coarse_pureUD_final") %>% select(Language, Surp, Pars) %>% rename(SurpGround = Surp) %>% rename(ParsGround = Pars) %>% mutate(EffGround = ParsGround + 0.9*SurpGround) %>% group_by(Language)
data = merge(dataBaseline, dataGround, by=c("Language"))

data$Eff = data$Pars + 0.9*data$Surp

u = data %>% group_by(Language) %>% summarise(BetterSurp = sum(Surp > SurpGround, na.rm=TRUE), WorseSurp = sum(Surp <= SurpGround, na.rm=TRUE), TotalSurp = BetterSurp+WorseSurp, BetterFracSurp = BetterSurp/TotalSurp, BetterPars = sum(Pars > ParsGround, na.rm=TRUE), WorsePars = sum(Pars <= ParsGround, na.rm=TRUE), TotalPars = BetterPars+WorsePars, BetterFracPars = BetterPars/TotalPars, BetterEff = sum(Eff > EffGround, na.rm=TRUE), WorseEff = sum(Eff <= EffGround, na.rm=TRUE), TotalEff = BetterEff+WorseEff, BetterFracEff = BetterEff/TotalEff )


pValuesPars_bin = c()
pValuesPars_t = c()

pValuesSurp_bin = c()
pValuesSurp_t = c()

pValuesEff_bin = c()
pValuesEff_t = c()

sink("per-language-statistics-pureUD-ground.tex")
for(i in (1:51)) {
	language = u$Language[[i]]
	#cat(language, "\n")

        testSurp_bin = binom.test(u$BetterSurp[[i]], u$TotalSurp[[i]], alternative="greater")
	pValuesSurp_bin = c(pValuesSurp_bin, testSurp_bin$p.value)
        v = data %>% filter(Language == language)
        testSurp_t = t.test(v$Surp, mu=mean(v$SurpGround), alternative="greater")
	pValuesSurp_t = c(pValuesSurp_t, testSurp_t$p.value)
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
        cat(paste("\\num{", format.pval(testSurp_t$p.value, digits=3), "}", sep=""))
        cat(" & ")
        cat(paste("\\num{", format.pval(testPars_t$p.value, digits=3), "}", sep=""))
        cat(" & ")
        cat(paste("$", round(testSurp_bin$estimate,2), "$ & $", paste("[", round(testSurp_bin$conf.int[[1]],2), ",", round(testSurp_bin$conf.int[[2]],2), "]", "$", sep=""), sep=""))
        cat(paste(" & ", "\\num{", format.pval(testSurp_bin$p.value, digits=3), "}", sep=""))
        cat(" & ")
        cat(paste("$", round(testPars_bin$estimate,2), "$ & $", paste("[", round(testPars_bin$conf.int[[1]],2), ",", round(testPars_bin$conf.int[[2]],2), "]", "$", sep=""), sep=""))
        cat(" & ")
        cat("\\num{", format.pval(testPars_bin$p.value, digits=3),"}", sep="")

#        cat(paste(round(testSurp_t$estimate,2), paste("[", round(testSurp_t$conf.int[[1]],2), ",", round(testSurp_t$conf.int[[2]],2), "]", sep=""), format.pval(testSurp_t$p.value, digits=3), sep=" & "))
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

u$pValuesSurp_bin = pValuesSurp_bin
u$pValuesSurp_t = pValuesSurp_t


mean(u$pValuesPars_t < 0.05)
mean(u$pValuesSurp_t < 0.05)
mean(u$pValuesPars_t < 0.025 | u$pValuesSurp_t < 0.025)
mean(u$pValuesEff_t < 0.05)

# Hochberg's step-up procedure
parse = sort(u$pValuesPars_t)
limit = 0.05/(51-(1:51)+1)
mean(parse <= limit)
# TODO Czech missing data for parseability!? doesn't come out at 100%

surp = sort(u$pValuesSurp_t)
limit = 0.05/(51-(1:51)+1)
mean(surp <= limit)

either = sort(pmin(u$pValuesSurp_t, u$pValuesPars_t))*2
limit = 0.05/(51-(1:51)+1)
mean(either <= limit)

# TODO Czech missing data!? doesn't come out at 100%

mean(u$pValuesPars_bin < 0.05)
mean(u$pValuesSurp_bin < 0.05)
mean(u$pValuesPars_bin < 0.025 | u$pValuesSurp_bin < 0.025)
mean(u$pValuesEff_bin < 0.05)

# Hochberg's step-up procedure
parse = sort(u$pValuesPars_bin)
limit = 0.05/(51-(1:51)+1)
mean(parse <= limit)
# TODO Czech missing data for parseability!? doesn't come out at 100%

surp = sort(u$pValuesSurp_bin)
limit = 0.05/(51-(1:51)+1)
mean(surp <= limit)

either = sort(pmin(u$pValuesSurp_bin, u$pValuesPars_bin))*2
limit = 0.05/(51-(1:51)+1)
mean(either <= limit)


 








