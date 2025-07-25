# KAN-LSTM with Hierarchical Attention for Option Price Forecasting

This project was developed for the **Stat-a-thon**, a statistics hackathon organized by **VIT Vellore**. We introduce the Temporal-KAN-LSTM with Option-specific Attention (TKLA), a novel hybrid neural architecture for forecasting the price of financial derivatives.

-----

## My Role and Contributions

As a key member of the team, my responsibilities were centered on the technical development and implementation of our proposed model.

  * [cite\_start]**Architectural Design**: I was responsible for conceptualizing and designing the novel **TKLA (Temporal-KAN-LSTM with Option-specific Attention)** architecture[cite: 7, 23]. [cite\_start]This involved synergistically combining Kolmogorov-Arnold Networks (KANs) with Long Short-Term Memory (LSTM) networks to address the dual challenges of capturing non-linear relationships and modeling complex temporal dependencies in financial data[cite: 8, 9].

  * **Sole Implementer & Coder**: I single-handedly undertook the entire coding and implementation of the architecture. This included:

      * [cite\_start]Developing the KAN-based feature extraction module using learnable B-spline activation functions[cite: 24, 75].
      * [cite\_start]Building the multi-scale LSTM component to process short-term, mid-range, and long-term temporal patterns in parallel[cite: 25, 109].
      * [cite\_start]Integrating an option-specific context module and a hierarchical attention mechanism to guide the model's focus[cite: 27, 28].

-----

## Project Abstract

[cite\_start]This research introduces the Temporal-KAN-LSTM with Option-specific Attention (TKLA), a novel neural architecture for financial derivatives pricing that synergistically combines Kolmogorov-Arnold Networks (KANs) with Long Short-Term Memory (LSTM) networks[cite: 8]. [cite\_start]Our model addresses the dual challenge of capturing non-linear relationships between financial indicators and modeling complex temporal dependencies in market data[cite: 9].

-----

## Model Architecture

[cite\_start]The TKLA architecture is a multi-component system designed for robust option price forecasting[cite: 68].

1.  [cite\_start]**Input Representation**: Preprocesses market and technical indicators[cite: 92].
2.  [cite\_start]**KAN-Based Feature Extraction**: Employs Kolmogorov-Arnold Networks with learnable B-spline functions on edges to capture complex, non-linear relationships between financial indicators without restrictive assumptions[cite: 24, 74, 87].
3.  [cite\_start]**Multi-scale Temporal Processing**: Uses a parallel set of short-term, medium-term, and long-term LSTM networks to model market dynamics across different time horizons simultaneously[cite: 25, 109].
4.  [cite\_start]**Option Context Encoding**: A dedicated module processes option-specific characteristics like strike price and time to maturity to create a context vector[cite: 28, 71].
5.  [cite\_start]**Hierarchical Attention Integration**: A two-level attention mechanism (feature-level and temporal-level) uses the option context vector to weigh the importance of different financial indicators and time steps[cite: 27, 130]. [cite\_start]This allows the model to adapt its focus based on the specific option being priced[cite: 143].
6.  [cite\_start]**Forecast Output**: A final prediction module combines the attention-weighted temporal features and the option context to generate the price forecast[cite: 105, 146].

[cite\_start]*Figure 1: Overall architecture showing flow from input features through feature extraction and multi-scale processing to final prediction via attention mechanisms[cite: 106].*

-----

## Proof of Work

My contributions are documented and verifiable through the following:

  * **Research Paper**: The submitted paper, "KAN-LSTM with Hierarchical Attention for Option Price Forecasting: A Hybrid Neural Architecture for Financial Derivatives," details the complete architecture that I designed and implemented.
  * **Codebase**: The complete source code for the TKLA model, which I developed independently, serves as the primary tangible proof of my work.
