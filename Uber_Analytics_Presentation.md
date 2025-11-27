# Customer Satisfaction for Transport Service Application – Uber
## Data Engineering Project Presentation

**Team DataCraft**
- Roupyajay Bhattacharya
- Umesh BP
- Madipally Bhagath Chandra
- Ninad Phadnis

**IISc Bangalore - M.Tech Program**

---

## Slide 1: Problem Statement

### Business Challenges
- **High Customer Churn Rate** - Users switching to competitors
- **Poor Customer Experience** - Long wait times and service inconsistencies
- **Driver Availability Issues** - Inefficient allocation leading to cancellations
- **Increasing Cancellations** - Both customer and driver-initiated

### Business Impact
- Revenue loss from incomplete rides
- Damaged brand reputation
- Competitive pressure from Lyft, Ola, and local services
- Need for actionable, data-driven insights

---

## Slide 2: Project Vision & Design Goals

### Core Objectives
**Scalability**
- Handle growing data volumes (150K → 100M+ records)
- Horizontal scaling with distributed computing

**Low Latency Processing**
- Near real-time ride completion predictions
- Fast ETL pipelines for operational insights

**Comprehensive Analytics**
- Real-time & batch data ingestion
- Predictive modeling for proactive interventions
- Dimensional modeling for business intelligence

---

## Slide 3: Technology Stack & Architecture

### Platform Components
**Apache Spark**
- 4GB driver/executor memory
- 200 shuffle partitions for optimal performance
- 2 executor cores for parallel processing

**Data Pipeline**
- **Source**: ncr_ride_bookings.csv (150K records)
- **Processing**: PySpark ETL with distributed computing
- **Storage**: Dimensional data warehouse model
- **ML Framework**: Spark MLlib for scalable machine learning

**Visualization & Analysis**
- Pandas, Matplotlib, Seaborn for EDA
- H2O AutoML for advanced modeling

---

## Slide 4: Data Pipeline & ETL Process

### ETL Workflow
**1. Data Ingestion**
- CSV batch upload with schema validation
- Automatic type casting and null handling

**2. Data Transformation**
- Timestamp creation (Date + Time merge)
- Feature engineering: `HourOfDay`, `DayOfWeek`
- Cancellation flag computation (`is_cancelled`)

**3. Data Quality**
- Null value imputation (0 for cancelled rides)
- Success rate calculation: **Ride Completion Rate**
- Acceptance rate: **Ride Acceptance Rate**

**4. Key Metrics Computed**
- Average booking value by hour
- Ride count distribution by time
- Cancellation rates by vehicle type

---

## Slide 5: Dimensional Data Model

### Star Schema Design

**Fact Table: `facts_rides`**
- Customer ID, Vehicle Type, Locations
- Avg VTAT, Avg CTAT, Ride Distance
- Timestamp, HourOfDay
- **Target**: RideCompleted (1=completed, 0=not completed)

**Dimension Tables**
- **dim_customer**: Customer ID, Customer Rating
- **dim_vehicle**: Vehicle Type (Go Mini, Auto, Sedan, etc.)
- **dim_location**: Pickup Location, Drop Location
- **dim_driver**: Driver Ratings

### Benefits
- Optimized for analytical queries
- Supports business intelligence reporting
- Scalable for growing data volumes

---

## Slide 6: Exploratory Data Analysis - Key Findings

### Booking Status Distribution
- Majority rides **Completed** successfully
- Significant cancellations identified (Customer: **~10%**, Driver: **~5%**)
- "No Driver Found" cases indicate supply issues

### Temporal Patterns
**Peak Hours Analysis**
- Morning rush: 7-9 AM (commute to work)
- Evening rush: 5-8 PM (return commutes)
- Late night spikes: Premium/airport rides

**Average Booking Value**
- Higher values during late night hours
- Lower values during mass transit hours
- Surge pricing effectiveness visible

### Vehicle Type Insights
- **Go Mini** most popular (33.3% of rides)
- **Auto** second most common (20%)
- Premium vehicles (Sedan, XL) show lower volume but higher value

---

## Slide 7: Cancellation Analysis

### Customer Cancellation Reasons (Top 5)
1. **Driver took too long to arrive**
2. **Changed destination**
3. **Booked by mistake**
4. **Found alternative transportation**
5. **Price too high**

### Driver Cancellation Patterns
- Primarily due to customer no-show or location issues
- Varies by vehicle type and time of day

### Critical Insights
- Wait time directly correlates with cancellation
- Vehicle Type vs Booking Status shows completion rates vary significantly
- High cancellation rates for certain vehicle types suggest supply-demand mismatch

---

## Slide 8: Machine Learning Pipeline Architecture

### Feature Engineering
**String Indexing** (Categorical → Numerical)
- Vehicle Type → VehicleType_Index
- Pickup Location → PickupLocation_Index (180+ locations)
- Drop Location → DropLocation_Index (190+ locations)

**Vector Assembly**
- Combined features: Avg VTAT, Avg CTAT, Ride Distance, Indexed categories
- Final feature vector: 6-dimensional

**Data Leakage Prevention**
- Removed post-event features: Payment Method, Driver Ratings, Customer Rating, Booking Value
- Only pre-ride features used for prediction

### Train/Test Split
- 80% training data (~120K records)
- 20% test data (~30K records)
- Stratified split with seed=42 for reproducibility

---

## Slide 9: Model Performance Comparison

### Three ML Models Evaluated

| Model | Accuracy | RMSE | Training Time | Best For |
|-------|----------|------|---------------|----------|
| **RandomForest** | **95.57%** | 0.2105 | 24s | Best accuracy-performance balance |
| **GBTClassifier** | 95.42% | 0.2140 | 32s | Slightly higher accuracy potential |
| **LogisticRegression** | 94.83% | 0.2273 | <1s | Ultra-fast baseline, real-time systems |

### Model Hyperparameters (RandomForest - Best Model)
- **numTrees**: 50 (optimal accuracy vs training time)
- **maxDepth**: 10 (prevents overfitting)
- **maxBins**: 200 (handles 180+ pickup locations)
- **seed**: 42 (reproducibility)

### Key Success Factors
- Distributed training across Spark executors
- Parallel tree construction (50 trees simultaneously)
- Efficient feature indexing and vector assembly

---

## Slide 10: Advanced Modeling - H2O AutoML

### Automated Machine Learning Integration
**H2O Sparkling Water**
- Integrated H2O AutoML with Spark pipeline
- Automated hyperparameter tuning
- Ensemble model selection

**AutoML Configuration**
- max_models: 20 (explores diverse algorithms)
- 5-fold cross-validation
- Automatic feature engineering

**Benefits**
- Leaderboard ranking of all models
- Best model selection based on validation performance
- Saved model for production deployment

### Scalability Advantage
- H2O distributed processing
- Handles larger datasets (100M+ records)
- Production-ready model artifacts

---

## Slide 11: Business Insights & Recommendations

### Operational Improvements
**1. Driver Allocation Optimization**
- Deploy more drivers during peak hours (7-9 AM, 5-8 PM)
- Focus on high-cancellation pickup locations
- Balance vehicle type availability by demand patterns

**2. Customer Experience Enhancement**
- Reduce wait times (primary cancellation reason)
- Implement dynamic pricing alerts
- Proactive ETA updates to reduce cancellations

**3. Predictive Interventions**
- Use 95.57% accurate model to identify at-risk bookings
- Real-time alerts for potential cancellations
- Driver incentives for high-risk rides

### Revenue Optimization
- Target premium hours (late night) with better driver coverage
- Optimize vehicle mix based on demand patterns
- Reduce cancellation-related revenue loss by 10-15%

---

## Slide 12: Conclusions & Future Work

### Project Achievements
✅ **Scalable ETL Pipeline** - Processes 150K records efficiently  
✅ **Dimensional Data Model** - Enables comprehensive analytics  
✅ **High-Accuracy ML Models** - 95.57% ride completion prediction  
✅ **Actionable Insights** - Identified key cancellation drivers  
✅ **Production-Ready Architecture** - Spark + H2O integration  

### Technical Highlights
- **Near-linear scalability**: Current architecture supports 100M+ records
- **Fast training**: 3 models trained in <60 seconds total
- **Reproducible**: Seed-based consistency for scientific rigor
- **Production deployment**: Serializable pipeline models

### Future Enhancements
**Phase 1 (Short-term)**
- Real-time streaming with Kafka integration
- A/B testing framework for model versions
- MLflow for model versioning and tracking

**Phase 2 (Long-term)**
- Deep learning models (LSTMs for time series)
- Geographic clustering for location-based optimization
- Multi-city expansion with federated learning
- Customer lifetime value prediction

---

## Thank You!

**Questions?**

**Contact:**
- Team DataCraft
- IISc Bangalore M.Tech Program
- Project Repository: [Include GitHub link if applicable]
