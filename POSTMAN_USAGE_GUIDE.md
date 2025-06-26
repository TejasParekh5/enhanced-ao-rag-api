# Postman Collection Usage Guide

## Overview
This guide provides instructions for using the **Optimized AO RAG API - Complete Collection** with Postman to test and interact with the AO RAG API endpoints.

## Prerequisites
1. **API Server Running**: Ensure the optimized API is running on `http://localhost:5001`
2. **Postman Installed**: Download from [postman.com](https://www.postman.com/)
3. **Collection Imported**: Import `Enhanced_AO_API_Postman_Collection.json`
4. **Test Data Available**: Reference `postman_test_data.json` for sample values

## Quick Start

### 1. Import the Collection
1. Open Postman
2. Click **Import** → **File** → Select `Enhanced_AO_API_Postman_Collection.json`
3. The collection will appear in your Collections tab

### 2. Verify API is Running
**First, run the Health Check:**
- Select **"Health Check"** request
- Click **Send**
- Should return: `{"status": "healthy", "system_initialized": true, ...}`

### 3. Get System Overview
**Run the Statistics endpoint:**
- Select **"System Statistics"** request
- Click **Send**
- View total AOs, applications, and risk metrics

## Endpoint Testing Guide

### 🏥 Health Check
**Purpose**: Verify API status and initialization
- **Method**: GET
- **URL**: `http://localhost:5001/health`
- **Expected Response Time**: < 200ms
- **Use Case**: System monitoring, startup verification

### 📊 System Statistics
**Purpose**: Get system-wide metrics and overview
- **Method**: GET
- **URL**: `http://localhost:5001/stats`
- **Expected Response Time**: < 300ms
- **Use Case**: Dashboard data, reporting, system overview

### 🔍 Search Endpoints

#### High Risk Applications
- **Purpose**: Find AOs with high-risk applications
- **Body**: `{"query": "high risk applications", "top_k": 5}`
- **Expected**: Ranked results with similarity scores

#### Department Search
- **Purpose**: Find AOs in specific departments
- **Body**: `{"query": "Identity Access Management", "top_k": 10}`
- **Expected**: AOs from relevant security departments

#### Critical Vulnerabilities
- **Purpose**: Identify AOs with critical security issues
- **Body**: `{"query": "critical vulnerabilities", "top_k": 3}`
- **Expected**: AOs with highest vulnerability counts

### 🎯 AO Suggestions (Main Analysis)

#### Comprehensive Analysis (With AI)
- **Purpose**: Get detailed security analysis with AI enhancement
- **Body**:
  ```json
  {
    "ao_name": "Alice Singh",
    "query": "What should I prioritize first?",
    "use_llm": true
  }
  ```
- **Expected Response Time**: < 3 seconds
- **Use Case**: Detailed security planning, executive reporting

#### Fast Analysis (Without AI)
- **Purpose**: Quick analysis using optimized algorithms only
- **Body**:
  ```json
  {
    "ao_name": "Deepa Nair",
    "query": "What are the main security risks?",
    "use_llm": false
  }
  ```
- **Expected Response Time**: < 1 second
- **Use Case**: Quick assessments, bulk analysis

#### Error Handling Test
- **Purpose**: Test system resilience with invalid input
- **Body**: `{"ao_name": "Invalid Name", "query": "Test error handling"}`
- **Expected**: Helpful error message with suggestions

## Sample AO Names (From Test Data)
- Alice Singh
- Deepa Nair
- Pooja Reddy
- Nisha Sharma
- Lata Menon
- Sneha Joshi
- Sonal Desai
- Priya Kapoor

## Common Request Patterns

### Basic Workflow
1. **Health Check** → Verify system is ready
2. **Statistics** → Get system overview
3. **Search** → Find relevant AOs
4. **Suggestions** → Get detailed analysis

### Performance Testing
1. Run **Health Check** multiple times (should be consistently fast)
2. Time the **Search** requests (should be < 500ms)
3. Compare **Suggestions** with/without LLM (AI takes longer)

### Error Testing
1. Test with **invalid AO names**
2. Test with **empty queries**
3. Test with **malformed JSON**

## Response Analysis

### Suggestions Response Structure
```json
{
  "success": true,
  "suggestions": {
    "ao_information": {
      "basic_info": {...},
      "security_metrics": {...},
      "vulnerability_breakdown": {...}
    },
    "security_analysis": {
      "overall_security_posture": "🔴 CRITICAL",
      "security_score": 25,
      "critical_concerns": [...],
      "risk_assessment": "..."
    },
    "priority_recommendations": [...],
    "action_items": {...},
    "compliance_guidance": {...},
    "comparative_analysis": {...},
    "risk_mitigation": {...}
  }
}
```

### Search Response Structure
```json
{
  "success": true,
  "query": "high risk applications",
  "results": [
    {
      "ao_name": "Alice Singh",
      "similarity_score": 0.87,
      "rank": 1,
      "risk_score": "4.44",
      "critical_vulnerabilities": "27",
      "applications": [...],
      "departments": [...],
      "searchable_text": "..."
    }
  ],
  "total_found": 5
}
```

## Advanced Usage Tips

### Environment Variables
Set up Postman environment variables:
- `base_url`: `http://localhost:5001`
- `ao_name`: `Alice Singh`
- `search_query`: `high risk applications`

### Collection Variables
The collection includes a `base_url` variable you can modify if running on different host/port.

### Automation
1. Use **Collection Runner** to test all endpoints sequentially
2. Set up **Pre-request Scripts** for dynamic data
3. Use **Tests** tab to add assertions

### Monitoring
1. Set up **Postman Monitors** for health checks
2. Create **Performance Tests** with multiple iterations
3. Use **Newman** for CI/CD integration

## Troubleshooting

### Common Issues
1. **"Connection refused"**: API server not running
2. **"RAG system not initialized"**: Wait for startup to complete
3. **"AO not found"**: Use exact names from test data
4. **Slow responses**: First run takes longer due to cache building

### Performance Expectations
- **Health**: < 200ms
- **Stats**: < 300ms  
- **Search**: < 500ms
- **Suggestions (no AI)**: < 1s
- **Suggestions (with AI)**: < 3s

### Debug Tips
1. Check **Console** tab for detailed logs
2. Use **Response** tab to inspect full JSON
3. Enable **SSL certificate verification** if using HTTPS
4. Check **Headers** for content-type and response codes

---

**Happy Testing! 🚀**

For more details, see:
- `COMPLETE_SYSTEM_DOCUMENTATION.md` - Full system documentation
- `ENDPOINT_TESTING_REPORT.md` - Comprehensive testing results
- `OPTIMIZATION_REPORT.md` - Performance improvements guide
