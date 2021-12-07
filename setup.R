options(htmltools.dir.version = FALSE)
knitr::opts_chunk$set(
  fig.width= 9,
  fig.height= 3.5,
  fig.retina= 3,
  out.width = "100%",
  cache = FALSE,
  echo = TRUE,
  message = FALSE,
  warning = FALSE
)

htmltools::tagList(
  xaringanExtra::use_clipboard(
    button_text = "<i class=\"fa fa-clipboard\"></i>",
    success_text = "<i class=\"fa fa-check\" style=\"color: #90BE6D\"></i>",
    error_text = "<i class=\"fa fa-times-circle\" style=\"color: #F94144\"></i>"
  ),
  rmarkdown::html_dependency_font_awesome(),
  xaringanExtra::use_tachyons(minified = TRUE),
  xaringanExtra::use_tile_view(),
  xaringanExtra::use_extra_styles(
    hover_code_line = TRUE,
    mute_unhighlighted_code = TRUE
  ),
  xaringanExtra::use_freezeframe(),
  xaringanExtra::use_scribble(
    pen_color = '#000000'
  )
)
