library(shiny)
library(data.table)
library(ggplot2)
library(plotly)
library(readr)
library(caret)
library(xgboost)
library(e1071)
library(randomForest)
library(shinyWidgets)

# === Load models ===
load_models <- function() {
  list(
    le = readRDS("label_encoder.rds"),
    scaler = readRDS("scaler_model.rds"),
    linear = readRDS("linear_regression_model.rds"),
    rf = readRDS("random_forest_model.rds"),
    xgb = xgb.load("xgboost_model.model"),
    svm = readRDS("svm_model.rds")
  )
}
models <- load_models()

# === Feature sets ===
regression_features <- c(
  "perimeter_mean", "compactness_mean", "area_mean", "concavity_mean",
  "concavity_se", "perimeter_se", "radius_worst"
)

classification_features <- c(
  "radius_mean", "perimeter_mean", "compactness_mean", "area_mean", "concavity_mean",
  "concavity_se", "perimeter_se", "radius_worst"
)

all_features <- c(
  "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
  "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
  "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
  "smoothness_se", "compactness_se", "concavity_se", "concave_points_se", "symmetry_se",
  "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
  "smoothness_worst", "compactness_worst", "concavity_worst", "concave_points_worst",
  "symmetry_worst", "fractal_dimension_worst"
)

# === Predict functions ===
predict_radius_mean <- function(data_row) {
  input_values <- as.numeric(unlist(data_row[regression_features]))
  input <- as.data.frame(t(input_values))
  colnames(input) <- regression_features
  input_matrix <- xgb.DMatrix(data.matrix(input))
  
  pred1 <- predict(models$linear, input)
  pred2 <- predict(models$rf, input)
  pred3 <- predict(models$xgb, input_matrix)
  
  radius_pred <- mean(c(pred1, pred2, pred3))
  c(radius_pred = radius_pred, min_pred = min(pred1, pred2, pred3), max_pred = max(pred1, pred2, pred3))
}

classify_tumor <- function(radius_pred, data_row) {
  input_data <- c(radius_mean = radius_pred,
                  sapply(classification_features[-1], function(f) as.numeric(data_row[[f]])))
  input <- as.data.frame(t(input_data))
  colnames(input) <- classification_features
  input_scaled <- predict(models$scaler, input)
  pred <- predict(models$svm, input_scaled, probability = TRUE)
  prob <- attr(pred, "probabilities")
  label <- ifelse(as.character(pred) == "1", "M", "B")
  confidence <- prob[1, as.character(pred)]
  list(label = label, confidence = confidence)
}

# === Shiny App ===
ui <- fluidPage(
  titlePanel("Breast Cancer Prediction"),
  tabsetPanel(
    # TAB 1 - Predict Radius
    tabPanel("Predict Radius Mean",
             fluidRow(
               lapply(regression_features, function(f) {
                 column(3, numericInput(f, label = f, value = 0))
               })
             ),
             actionButton("predict_btn", "Predict Radius"),
             verbatimTextOutput("radius_result")
    ),
    
    # TAB 2 - Classify Tumor
    tabPanel("Classify Tumor",
             h4("Predicted Radius Mean (from Tab 1):"),
             textOutput("radius_from_pred"),   # hiển thị radius đã dự đoán
             br(),
             fluidRow(
               lapply(classification_features[-1], function(f) {
                 column(3, numericInput(paste0("cls_", f), label = f, value = 0))
               })
             ),
             actionButton("classify_btn", "Classify"),
             verbatimTextOutput("classify_result")
    ),
    
    # TAB 3 - CSV Upload
    tabPanel("CSV Upload",
             fileInput("csv_file", "Upload CSV", accept = ".csv"),
             dataTableOutput("table"),
             plotlyOutput("pie_chart"),
             plotlyOutput("hist_chart"),
             plotlyOutput("scatter_plot")
    )
  )
)

server <- function(input, output, session) {
  # Reactive để lưu giá trị radius dự đoán
  radius_pred_val <- reactiveVal(NULL)
  
  # === TAB 1: Predict Radius ===
  observeEvent(input$predict_btn, {
    data_row <- lapply(regression_features, function(f) input[[f]])
    names(data_row) <- regression_features
    radius_info <- predict_radius_mean(data_row)
    radius_pred_val(radius_info["radius_pred"])  # lưu radius_mean
    
    output$radius_result <- renderPrint({
      cat("Predicted Radius Mean Range:",
          paste0("[", round(radius_info["min_pred"],2), " - ", round(radius_info["max_pred"],2), "]\n"))
      cat("Final Predicted Radius Mean:", round(radius_info["radius_pred"], 2), "\n")
    })
  })
  
  # === TAB 2: Classify Tumor ===
  output$radius_from_pred <- renderText({
    val <- radius_pred_val()
    if (is.null(val)) {
      "Chưa có kết quả dự đoán, vui lòng sang Tab 'Predict Radius Mean' để dự đoán trước."
    } else {
      paste("Radius Mean =", round(val, 2))
    }
  })
  
  observeEvent(input$classify_btn, {
    req(radius_pred_val())  # cần radius từ tab 1
    data_row <- lapply(classification_features[-1], function(f) input[[paste0("cls_", f)]])
    names(data_row) <- classification_features[-1]
    
    result <- classify_tumor(radius_pred_val(), data_row)
    output$classify_result <- renderPrint({
      cat("Diagnosis:", result$label, " (", round(result$confidence * 100, 2), "%)\n")
    })
  })
  
  # === TAB 3: CSV Upload ===
  observeEvent(input$csv_file, {
    req(input$csv_file)
    df <- fread(input$csv_file$datapath)
    df[] <- lapply(df, as.numeric)
    
    results <- lapply(1:nrow(df), function(i) {
      row <- as.list(df[i, ])
      tryCatch({
        radius_info <- predict_radius_mean(row)
        cls <- classify_tumor(radius_info["radius_pred"], row)
        list(
          Index = i,
          RadiusRange = paste0("[", round(radius_info["min_pred"],2), " - ", round(radius_info["max_pred"],2), "]"),
          Diagnosis = cls$label,
          Confidence = round(cls$confidence * 100, 2)
        )
      }, error = function(e) list(Index = i, Error = e$message))
    })
    
    results_df <- rbindlist(results, fill = TRUE)
    output$table <- renderDataTable(results_df)
    
    output$pie_chart <- renderPlotly({
      req(results_df$Diagnosis)
      counts <- as.data.frame(table(results_df$Diagnosis))
      counts$Label <- paste(counts$Var1, "(", counts$Freq, ")")
      plot_ly(counts, labels = ~Label, values = ~Freq, type = 'pie') %>%
        layout(title = 'Diagnosis Distribution')
    })
    
    output$hist_chart <- renderPlotly({
      req(results_df$RadiusRange)
      radius_mids <- sapply(results_df$RadiusRange, function(r) {
        nums <- as.numeric(unlist(regmatches(r, gregexpr("[0-9.]+", r))))
        mean(nums)
      })
      plot_ly(x = radius_mids, type = "histogram", nbinsx = 20,
              marker = list(line = list(width = 1, color = 'black')),
              name = "Predicted Radius Mean") %>%
        layout(title = "Predicted Radius Mean Distribution")
    })
    
    output$scatter_plot <- renderPlotly({
      plot_ly(results_df, x = ~Index, y = ~Confidence,
              type = 'scatter', mode = 'markers', color = ~Diagnosis,
              colors = c("B" = "blue", "M" = "red"),
              marker = list(size = 10, line = list(width = 1))) %>%
        layout(title = "Radius Mean vs Confidence")
    })
  })
}

shinyApp(ui, server)
