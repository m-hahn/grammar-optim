



library(lme4)

library(tidyr)
library(dplyr)
library(ggplot2)




depl = read.csv("/home/user/CS_SCR/deps/dependency_length/total_summary_funchead_coarse.tsv", sep="\t")# %>% rename(Quality=AverageLength)
library(tidyr)
library(dplyr)


depl = depl %>% filter(Type %in% c("manual_output_funchead_coarse_depl", "manual_output_funchead_langmod_coarse_best_balanced","manual_output_funchead_RANDOM","manual_output_funchead_two_coarse_lambda09_best_balanced","manual_output_funchead_two_coarse_parser_best_balanced", "REAL_REAL"))

depl$Temperature = NULL
depl$Counter = NULL
depl$OriginalLoss = NULL
depl$OriginalCounter = NULL


bestDepL = read.csv("../strongest_models/best-depl.csv") %>% select(Language, Type, Model)
bestLangmod = read.csv("../strongest_models/best-langmod-best-balanced.csv") %>% select(Language, Type, Model)
bestParse = read.csv("../strongest_models/best-parse-best-balanced.csv") %>% select(Language, Type, Model)
bestEff = read.csv("../strongest_models/best-two-lambda09-best-balanced.csv") %>% select(Language, Type, Model)

bestModels = rbind(bestDepL, bestLangmod, bestParse, bestEff)

deplRand = depl %>% filter(Type == "manual_output_funchead_RANDOM")
deplReal = depl %>% filter(Type == "REAL_REAL")

depl = merge(depl, bestModels, by=c("Language", "Model", "Type"))
depl = rbind(depl, deplRand, deplReal)

depl2 = depl %>% group_by(Language, Type) %>% summarise(AverageLengthPerWord = mean(AverageLengthPerWord)) #%>% spread(Type, AverageLengthPerWord)

depl4 = depl2 %>% spread(Type, AverageLengthPerWord)


meanPerLang = depl2 %>% group_by(Language) %>% summarise(MeanLength = mean(AverageLengthPerWord))
depl2 = merge(depl2, meanPerLang, by=c("Language"))
depl2 = depl2 %>% filter(!(Type %in% c("manual_output_funchead_langmod_coarse_best_balanced", "manual_output_funchead_two_coarse_parser_best_balanced")))

plot = ggplot(data=depl2, aes(x=AverageLengthPerWord-MeanLength, group=Type, fill=Type)) + geom_density(alpha=0.3)

depl3 = depl %>% filter(!(Type %in% c("manual_output_funchead_langmod_coarse_best_balanced", "manual_output_funchead_two_coarse_parser_best_balanced")))
depl3$Type = factor(depl3$Type, levels=c("manual_output_funchead_coarse_depl", "manual_output_funchead_two_coarse_lambda09_best_balanced", "REAL_REAL",  "manual_output_funchead_RANDOM"  ))
library(forcats)
depl3$TypeN = fct_recode(depl3$Type, DLM="manual_output_funchead_coarse_depl", Eff="manual_output_funchead_two_coarse_lambda09_best_balanced", Real="REAL_REAL", Baseline="manual_output_funchead_RANDOM")


library(ggrepel)
plot = ggplot(data=depl3, aes(x=TypeN, y=AverageLengthPerWord)) + geom_point() + geom_line(data=depl3 %>% group_by(TypeN) %>% summarise(AverageLengthPerWord = mean(AverageLengthPerWord)), aes(x=TypeN, y=AverageLengthPerWord, group=1)) + geom_text_repel(data=depl3 %>% filter(grepl("Real", TypeN)), aes(label=Language),hjust=0, vjust=0)


# David Robinson
# https://gist.github.com/dgrtwo
# Modifications by PoGibas
# https://stackoverflow.com/questions/52034747/plot-only-one-side-half-of-the-violin-plot
library(ggplot2)
library(dplyr)


"%||%" <- function(a, b) {
  if (!is.null(a)) a else b
}

geom_flat_violin <- function(mapping = NULL, data = NULL, stat = "ydensity",
                        position = "dodge", trim = TRUE, scale = "area",
                        show.legend = NA, inherit.aes = TRUE, ...) {
  layer(
    data = data,
    mapping = mapping,
    stat = stat,
    geom = GeomFlatViolin,
    position = position,
    show.legend = show.legend,
    inherit.aes = inherit.aes,
    params = list(
      trim = trim,
      scale = scale,
      ...
    )
  )
}

GeomFlatViolin <-
  ggproto("GeomFlatViolin", Geom,
          setup_data = function(data, params) {
            data$width <- data$width %||%
              params$width %||% (resolution(data$x, FALSE) * 0.9)

            # ymin, ymax, xmin, and xmax define the bounding rectangle for each group
            data %>%
              group_by(group) %>%
              mutate(ymin = min(y),
                     ymax = max(y),
                     xmin = x - width / 2,
                     xmax = x)
          },

          draw_group = function(data, panel_scales, coord) {
            # Find the points for the line to go all the way around
            data <- transform(data,
                              xmaxv = x,
                              xminv = x + violinwidth * (xmin - x))

            # Make sure it's sorted properly to draw the outline
            newdata <- rbind(plyr::arrange(transform(data, x = xminv), y),
                             plyr::arrange(transform(data, x = xmaxv), -y))

            # Close the polygon: set first and last point the same
            # Needed for coord_polar and such
            newdata <- rbind(newdata, newdata[1,])

            ggplot2:::ggname("geom_flat_violin", GeomPolygon$draw_panel(newdata, panel_scales, coord))
          },

          draw_key = draw_key_polygon,

          default_aes = aes(weight = 1, colour = "grey20", fill = "white", size = 0.5,
                            alpha = NA, linetype = "solid"),

          required_aes = c("x", "y")
)

depl3Baseline = depl3 %>% group_by(Language) %>% filter(TypeN == "Baseline") %>% summarise(AverageLengthPerWord = mean(AverageLengthPerWord)) %>% mutate(TypeN = "Baseline")
depl3 = rbind(depl3 %>% select(TypeN, Language, AverageLengthPerWord) %>% filter(TypeN != "Baseline"), depl3Baseline)

plot = ggplot(data=depl3, aes(x=TypeN, y=AverageLengthPerWord)) + geom_point() 
plot = plot + geom_line(data=depl3 %>% group_by(TypeN) %>% summarise(AverageLengthPerWord = mean(AverageLengthPerWord)), aes(x=TypeN, y=AverageLengthPerWord, group=1))
plot = plot + geom_line(data=depl3 %>% filter(Language == "English") %>% group_by(TypeN) %>% summarise(AverageLengthPerWord = mean(AverageLengthPerWord)), aes(x=TypeN, y=AverageLengthPerWord, group=1), alpha=0.5)
plot = plot + geom_line(data=depl3 %>% filter(Language == "Russian") %>% group_by(TypeN) %>% summarise(AverageLengthPerWord = mean(AverageLengthPerWord)), aes(x=TypeN, y=AverageLengthPerWord, group=1), alpha=0.5)
plot = plot + geom_line(data=depl3 %>% filter(Language == "Japanese") %>% group_by(TypeN) %>% summarise(AverageLengthPerWord = mean(AverageLengthPerWord)), aes(x=TypeN, y=AverageLengthPerWord, group=1), alpha=0.5)
plot = plot + geom_line(data=depl3 %>% filter(Language == "Arabic") %>% group_by(TypeN) %>% summarise(AverageLengthPerWord = mean(AverageLengthPerWord)), aes(x=TypeN, y=AverageLengthPerWord, group=1), alpha=0.5)
plot = plot + geom_flat_violin(data=depl3, aes(x=TypeN, y=AverageLengthPerWord, fill=TypeN))
plot = plot + geom_text_repel(data=depl3 %>% filter(Language %in% c("English", "Russian", "Japanese", "Arabic")) %>% filter( TypeN == "Real"), aes(label=Language, alpha=1.0),hjust=-1, vjust=0)
plot = plot + theme_bw()
plot = plot + theme(legend.position="none")
plot = plot + theme(legend.position="none")
plot = plot + xlab(NULL)
plot = plot + ylab("Mean Dependency Length")

ggsave(plot, file="figures/depl-violin.pdf")




mean(depl2$REAL_REAL < depl2$manual_output_funchead_coarse_depl)
mean(depl2$REAL_REAL < depl2$manual_output_funchead_langmod_coarse_best_balanced)
mean(depl2$REAL_REAL < depl2$manual_output_funchead_RANDOM, na.rm=TRUE)
mean(depl2$REAL_REAL < depl2$manual_output_funchead_two_coarse_parser_best_balanced, na.rm=TRUE)
mean(depl2$REAL_REAL < depl2$manual_output_funchead_two_coarse_lambda09_best_balanced, na.rm=TRUE)


