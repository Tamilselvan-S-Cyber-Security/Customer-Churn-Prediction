# Tab 6: AI Explanations
with tab6:
    st.subheader("AI-Powered Churn Analysis & Explanations")
    
    if st.session_state.model is not None and st.session_state.data is not None:
        # Introduction
        st.markdown("""
        ### AI-Powered Insights
        This tab provides intelligent explanations and insights about your churn prediction model, 
        helping you understand the factors driving customer churn and develop targeted retention strategies.
        """)
        
        # Create tabs for different types of explanations
        explain_tab1, explain_tab2, explain_tab3 = st.tabs([
            "Model Explanations", 
            "Customer Profile Analysis", 
            "Retention Recommendations"
        ])
        
        # Tab 1: Model Explanations
        with explain_tab1:
            st.markdown("### Understanding Your Churn Prediction Model")
            
            if st.session_state.metrics is not None and st.session_state.feature_importance is not None:
                # Generate model overview and insights
                model_overview = generate_model_overview(
                    st.session_state.model,
                    st.session_state.metrics,
                    st.session_state.feature_importance
                )
                
                # Display model summary
                st.markdown(f"#### {model_overview['model_name']} Model Summary")
                st.markdown(model_overview['summary'])
                
                # Display model performance metrics
                st.markdown("#### Model Performance")
                
                # Create metrics display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{model_overview['performance']['accuracy']:.2%}")
                with col2:
                    st.metric("Precision", f"{model_overview['performance']['precision']:.2%}")
                with col3:
                    st.metric("Recall", f"{model_overview['performance']['recall']:.2%}")
                with col4:
                    st.metric("F1 Score", f"{model_overview['performance']['f1']:.2%}")
                
                # Feature importance analysis
                st.markdown("#### Key Churn Factors")
                st.markdown(model_overview['business_insight'])
                
                # Generate detailed explanations of important features
                feature_explanations = get_detailed_feature_explanation(
                    st.session_state.model,
                    st.session_state.X_train.columns,
                    st.session_state.feature_importance,
                    top_n=5
                )
                
                # Display feature explanations
                st.markdown("#### Feature Importance Explanations")
                
                for i, (feature, details) in enumerate(feature_explanations.items()):
                    with st.expander(f"{i+1}. {feature} (Importance: {details['importance']:.4f})"):
                        st.markdown(f"**Impact Level:** {details['impact_level'].title()}")
                        st.markdown(details['explanation'])
                
                # Model strengths and limitations
                st.markdown("#### Model Strengths & Limitations")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Strengths**")
                    for strength in model_overview.get('model_strengths', ['Advanced pattern recognition']):
                        st.markdown(f"- {strength}")
                
                with col2:
                    st.markdown("**Limitations**")
                    for limitation in model_overview.get('model_limitations', ['Limited to patterns in historical data']):
                        st.markdown(f"- {limitation}")
                
                # How to use the model effectively
                st.markdown("#### How to Use This Model Effectively")
                st.markdown("""
                1. **Focus on top features**: Concentrate retention efforts on the highest importance factors
                2. **Target high probability churners**: Prioritize customers with the highest churn probability
                3. **Monitor performance**: Regularly check model metrics as customer behavior evolves
                4. **Balance precision and recall**: Adjust intervention thresholds based on your business priorities
                5. **Combine with domain knowledge**: Use model insights alongside your business expertise
                """)
            
            else:
                st.info("Please train a model first to access AI-powered model explanations.")
        
        # Tab 2: Customer Profile Analysis
        with explain_tab2:
            st.markdown("### Individual Customer Analysis")
            
            if st.session_state.model is not None and st.session_state.feature_importance is not None:
                # Option to select a customer from the dataset
                st.markdown("#### Select a customer to analyze")
                
                # Allow selecting a customer
                if 'CustomerID' in st.session_state.data.columns:
                    customer_id = st.selectbox(
                        "Select Customer ID", 
                        options=st.session_state.data['CustomerID'].unique()
                    )
                    
                    # Get the customer data
                    customer_data = st.session_state.data[st.session_state.data['CustomerID'] == customer_id]
                else:
                    # If no CustomerID, use row index
                    customer_idx = st.selectbox(
                        "Select Customer (by row index)", 
                        options=range(len(st.session_state.data))
                    )
                    
                    # Get the customer data
                    customer_data = st.session_state.data.iloc[[customer_idx]]
                
                if not customer_data.empty:
                    # Display basic customer information
                    st.markdown("#### Customer Profile")
                    
                    # Extract key demographic info if available
                    profile_cols = [col for col in customer_data.columns if col in 
                                  ['CustomerID', 'Gender', 'Age', 'Tenure', 'Contract', 
                                   'MonthlyCharges', 'TotalCharges']]
                    
                    if profile_cols:
                        st.dataframe(customer_data[profile_cols])
                    else:
                        st.dataframe(customer_data.iloc[:, :5])  # Show first 5 columns
                    
                    # Analyze customer profile
                    profile_analysis = analyze_customer_profile(
                        customer_data, 
                        st.session_state.feature_importance
                    )
                    
                    # Display profile insights
                    st.markdown("#### Key Insights")
                    for insight in profile_analysis['insights']:
                        st.markdown(f"- {insight}")
                    
                    # Make prediction for this customer
                    # Preprocess the customer data (simplified - in real app would need more complex preprocessing)
                    if st.session_state.target_column in customer_data.columns:
                        actual_churn = customer_data[st.session_state.target_column].iloc[0]
                        st.markdown(f"**Actual Churn Status:** {'Yes' if actual_churn else 'No'}")
                    
                    # Add a button to generate a detailed analysis
                    if st.button("Generate Detailed Analysis"):
                        with st.spinner("Analyzing customer profile..."):
                            # In a real implementation, this would use the Anthropic API for more detailed analysis
                            # For now, we'll use our rule-based insights
                            
                            st.markdown("#### Churn Risk Assessment")
                            
                            # For demonstration, we'll create a placeholder risk assessment
                            # In a real app, this would use the model to make an actual prediction
                            if 'Tenure' in customer_data.columns and 'Contract' in customer_data.columns:
                                tenure = customer_data['Tenure'].iloc[0]
                                contract = str(customer_data['Contract'].iloc[0]).lower()
                                
                                if tenure < 12 and 'month' in contract:
                                    risk_level = "High"
                                    risk_color = "🔴"
                                elif tenure < 24 and 'month' in contract:
                                    risk_level = "Medium"
                                    risk_color = "🟠"
                                else:
                                    risk_level = "Low" 
                                    risk_color = "🟢"
                                
                                st.markdown(f"**Churn Risk Level:** {risk_color} {risk_level}")
                            
                            # Generate personalized recommendation
                            st.markdown("#### Personalized Engagement Strategy")
                            st.markdown("""
                            1. **Targeted Communication**: Engage based on usage patterns and preferences
                            2. **Value Demonstration**: Highlight specific benefits most relevant to this customer
                            3. **Proactive Support**: Address potential issues before they trigger churn
                            4. **Customized Offers**: Develop offers aligned with the customer's specific needs
                            """)
            
            else:
                st.info("Please train a model first to access customer profile analysis.")
        
        # Tab 3: Retention Recommendations
        with explain_tab3:
            st.markdown("### Smart Retention Strategies")
            
            if st.session_state.model is not None:
                # Customer selection for retention recommendations
                st.markdown("#### Generate Personalized Retention Strategies")
                
                # Allow selecting a customer
                if 'CustomerID' in st.session_state.data.columns:
                    retention_customer_id = st.selectbox(
                        "Select Customer ID", 
                        options=st.session_state.data['CustomerID'].unique(),
                        key="retention_customer_id"
                    )
                    
                    # Get the customer data
                    retention_customer_data = st.session_state.data[st.session_state.data['CustomerID'] == retention_customer_id]
                else:
                    # If no CustomerID, use row index
                    retention_customer_idx = st.selectbox(
                        "Select Customer (by row index)", 
                        options=range(len(st.session_state.data)),
                        key="retention_customer_idx"
                    )
                    
                    # Get the customer data
                    retention_customer_data = st.session_state.data.iloc[[retention_customer_idx]]
                
                if not retention_customer_data.empty:
                    # Generate a simulated churn probability (in a real app, would use model prediction)
                    # For simplicity, we'll generate a random probability based on customer attributes
                    churn_prob = 0.5  # Default
                    
                    if 'Tenure' in retention_customer_data.columns:
                        tenure = retention_customer_data['Tenure'].iloc[0]
                        # Lower tenure generally means higher churn risk
                        if tenure < 12:
                            churn_prob += 0.3
                        elif tenure > 36:
                            churn_prob -= 0.3
                    
                    if 'Contract' in retention_customer_data.columns:
                        contract = str(retention_customer_data['Contract'].iloc[0]).lower()
                        # Month-to-month typically has higher churn
                        if 'month' in contract:
                            churn_prob += 0.2
                        elif 'two year' in contract:
                            churn_prob -= 0.2
                    
                    # Ensure probability is between 0 and 1
                    churn_prob = max(0.05, min(0.95, churn_prob))
                    
                    # Generate recommendations
                    recommendations = generate_retention_recommendations(
                        churn_prob,
                        retention_customer_data,
                        type(st.session_state.model).__name__
                    )
                    
                    # Display risk assessment
                    st.markdown("#### Churn Risk Assessment")
                    
                    # Create a gauge chart for churn probability
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=churn_prob * 100,
                        title={'text': "Churn Probability"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "red"},
                            'steps': [
                                {'range': [0, 30], 'color': "green"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ]
                        }
                    ))
                    
                    fig.update_layout(
                        height=250,
                        plot_bgcolor='#0E1117',
                        paper_bgcolor='#0E1117',
                        font_color='white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(f"**Risk Level:** {recommendations['risk_level']}")
                    st.markdown(recommendations['explanation'])
                    
                    # Display immediate actions
                    st.markdown("#### Recommended Immediate Actions")
                    if recommendations['immediate_actions']:
                        for i, action in enumerate(recommendations['immediate_actions']):
                            st.markdown(f"{i+1}. {action}")
                    else:
                        st.markdown("No immediate actions required. Continue standard engagement.")
                    
                    # Display strategic recommendations
                    st.markdown("#### Strategic Recommendations")
                    for i, strategy in enumerate(recommendations['strategic_recommendations']):
                        st.markdown(f"{i+1}. {strategy}")
                    
                    # Additional retention tools
                    st.markdown("#### Retention Tools")
                    
                    # Display different tools based on risk level
                    if recommendations['risk_level'] == "HIGH":
                        st.markdown("""
                        - **VIP Rescue Program**: Assign dedicated success manager for personalized attention
                        - **Exclusive Promotion**: Offer special time-limited discount or upgrade
                        - **Executive Outreach**: Schedule high-level contact for relationship building
                        - **Custom Package**: Design tailored solution for specific needs
                        """)
                    elif recommendations['risk_level'] == "MEDIUM":
                        st.markdown("""
                        - **Satisfaction Survey**: Identify and address specific pain points
                        - **Feature Education**: Ensure customer is utilizing all relevant features
                        - **Loyalty Rewards**: Provide additional value for continued loyalty
                        - **Usage Analysis**: Review usage patterns to suggest optimizations
                        """)
                    else:
                        st.markdown("""
                        - **Referral Program**: Leverage satisfaction to expand customer base
                        - **Upsell Opportunities**: Identify potential product/service expansions
                        - **Success Stories**: Share how similar customers are benefiting
                        - **Regular Check-ins**: Maintain engagement through periodic contact
                        """)
            
            else:
                st.info("Please train a model first to access retention recommendations.")
    
    else:
        st.info("Please load data and train a model to access AI-powered explanations.")
        st.markdown("""
        ### What You'll Get From AI Explanations
        
        Once you've trained a model, this tab will provide:
        
        1. **Model Explanations**: Understand what factors most influence customer churn
        2. **Customer Profile Analysis**: Get insights on individual customers and their churn risk
        3. **Retention Recommendations**: Receive personalized strategies to reduce churn
        
        Start by using the sample data, preprocessing it, and training a model in the previous tabs.
        """)