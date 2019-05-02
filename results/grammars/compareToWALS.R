
library(tidyr)
library(dplyr)
library(lme4)
library(ggplot2)
library(forcats)

# Note the complication that Serbian-Croatian is one language in WALS, and two languages in UD. In contrast, Hindi and Urdu are separate in both.

wals_path = "~/mhahn_files/stnfclasses/deps/writeup/language.csv"
wals = read.csv(wals_path)


wals = wals %>% mutate(obj_w_ = as.character(X83A.Order.of.Object.and.Verb))
wals = wals %>% mutate(lifted_mark_w_ = as.character(wals$X94))
wals = wals %>% mutate(acl_w_ = as.character(wals$X90A))
wals = wals %>% mutate(nmod_w_ = as.character(wals$X86))
wals = wals %>% mutate(lifted_case_w_ = as.character(wals$X85))
wals = wals %>% mutate(obl_w_ = as.character(wals$X84A))


wals = wals %>% mutate(obj_w = as.character(as.factor(case_when(obj_w_ ==  "2 VO" ~ "HD", obj_w_ == "1 OV" ~ "DH"))))
wals = wals %>% mutate(lifted_mark_w = as.character(case_when(lifted_mark_w_ == "1 Initial subordinator word" ~ "HD", lifted_mark_w_ == "2 Final subordinator word" ~ "DH")))
wals = wals %>% mutate(acl_w = as.character(case_when(acl_w_ == "1 Noun-Relative clause" ~ "HD", acl_w_ == "2 Relative clause-Noun" ~ "DH")))
wals = wals %>% mutate(nmod_w = as.character(as.factor(case_when(nmod_w_ == "2 Noun-Genitive" ~ "HD", nmod_w_ == "1 Genitive-Noun" ~ "DH"))))
wals = wals %>% mutate(lifted_case_w = as.character(as.factor(case_when(lifted_case_w_ %in% c("1 Postpositions" )  ~ "DH", lifted_case_w_ %in% c("2 Prepositions") ~ "HD"))))
wals = wals %>% mutate(obl_w = as.character(as.factor(case_when(obl_w_ %in% c("1 VOX","5 OVX" )  ~ "HD", obl_w_ %in% c("2 XVO", "4 OXV", "3 XOV") ~ "DH"))))


languages_wals_mapping = read.csv("languages-wals-mapping.csv")
wals$Language = NULL
wals$iso_code = NULL

wals = merge(languages_wals_mapping, wals, by=c("Name"), all.x=TRUE)


#ud = read.csv("../languages/languages-iso_codes.tsv")
#wals = merge(ud, wals, by=c("iso_code"), all.x=TRUE)
#
#wals = wals %>% select(Language, Name, iso_code, obj_w, lifted_mark_w, acl_w, obl_w, obj_w_, lifted_mark_w_, acl_w_, obl_w_)
#
#wals = wals %>% select(Language, Name, iso_code)
#write.csv(wals, file="languages-wals-mapping.csv")


ofInterest =  c("acl", "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "obl", "xcomp")
data = read.csv("../../grammars/manual_output_funchead_ground_coarse_final/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)
data = data %>% mutate(Language = fct_recode(Language, "Old_Church_Slavonic" = "Old", "Ancient_Greek" = "Ancient"))
data = data %>% mutate(DH_Weight = DH_Mean_NoPunct)
data = data %>% mutate(Order = case_when(DH_Weight > 0.0 ~ "DH", DH_Weight < 0.0 ~ "HD"))
data = data %>% select(Language, Dependency, Order)


wals = wals %>% select(Language, Name, iso_code, obj_w, lifted_case_w, lifted_mark_w, acl_w, obl_w, nmod_w, obj_w_, lifted_case_w_, lifted_mark_w_, acl_w_, obl_w_, nmod_w_)


getJudgD = function(d, dep) {
	d = d %>% filter(Dependency == dep)
	if((nrow(d)) == 0) {
		return("--")
	}
   return ((d$Order)[[1]])
}
getJudgW = function(w, dep) {
	if(nrow(w) == 0) {
		return("?")
	}
   return (w[[paste(dep, "w", sep="_")]])
}
getJudgWF = function(w, dep) {
	if(nrow(w) == 0) {
		return("?")
	}
    value = substr(w[[paste(dep, "w", "", sep="_")]], 1, 1)
    if(is.na(value)) {
	    value = "?"
    } else if(value == "") {
	    value = "?"
    }
   return(value)
}


getJudgWF2 = function(w, dep) {
	if(nrow(w) == 0) {
		return("?")
	}
	value = (w[[paste(dep, "w", sep="_")]])
	if(is.null(value)) {
		return("?")
	} else     if(is.na(value)) {
        value = substr(w[[paste(dep, "w", "", sep="_")]], 1, 1)
     if(is.na(value)) {
	     value = "?"
     } else  if(value == "") {
	    value = "?"
        } else {
		value = "*"
     }
     
     }
	if(grepl("No dominant", w[[paste(dep, "w", "", sep="_")]])) {
		value = "*"
	}
   return(value)
}


total_both = 0
disagreement=0

total_d = 0
total_w = 0
total_w_dom = 0

sink("comparison-table.tex")
for(language in sort(unique(wals$Language))) {
   d = data %>% filter(Language == language)
   w = wals %>% filter(Language == language)
   if(language == "Old_Church_Slavonic") {
	   language = "O.C.Slav."
   } else if(language == "Ancient_Greek") {
	   language = "Anc.Grk."
   }
   cat(gsub("_", "\\\\_", language), " & ")
   for(dep in c("obj", "lifted_case", "lifted_mark", "acl", "nmod", "obl")) {
	   jd = getJudgD(d, dep)
	   jw = getJudgWF2(w, dep)
	   if(jd != "--") {
		   total_d = total_d+1
		   if(jw != "?") {
			   total_w = total_w+1
			   if(jw %in% c("HD", "DH")) {
				   total_w_dom = total_w_dom + 1
			   }
		   }
	   }
	   if(jd %in% c("HD", "DH") && jw %in% c("HD", "DH")){
		   total_both = total_both+1
	   }
	   if(jd != jw && jd %in% c("HD", "DH") && jw %in% c("HD", "DH")) {
		   disagreement = disagreement+1
              cat("\\textit{", jd, "}", " & ", sep="")
              cat(jw, " & ")
	   } else{
               cat(jd, " & ")
               cat(jw, " & ")

	   }
   }
   cat("\\\\ \n")
}
cat()
sink()


cat("Agreement", 1-(disagreement)/total_both, "\n")
cat("WALS has some entry for ",total_w/total_d, "\n")
cat("WALS lists dominant order for ", total_w_dom/total_w, "\n")

