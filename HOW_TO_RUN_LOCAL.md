# 🚀 How to Run the PLC Logic Decompiler Web Application Locally

## 📋 **QUICK START GUIDE**

### **Prerequisites**
- ✅ Python 3.8+ (you have Python 3.12 in conda pyoccenv) 
- ✅ Conda environment already configured
- ✅ Windows machine (your current setup)

### **Step-by-Step Deployment Instructions**

## 🔧 **OPTION 1: Quick Start (Recommended)**

### **1. Open Command Prompt/Terminal**
```cmd
# Open Command Prompt as Administrator (recommended)
# OR use Windows Terminal
```

### **2. Navigate to Project Directory**
```cmd
cd "c:\Users\jjacominiprioli\OneDrive - North Carolina A&T State University\Articles\PLC logic decompiler"
```

### **3. Activate Conda Environment**
```cmd
conda activate pyoccenv
```

### **4. Install Required Web Dependencies**
```cmd
conda run -n pyoccenv pip install flask werkzeug jinja2 itsdangerous click markupsafe
```

### **5. Start the Flask Application**
```cmd
conda run -n pyoccenv python app.py
```

### **6. Access the Web Application**
- **Open your web browser**
- **Navigate to**: `http://127.0.0.1:5000` or `http://localhost:5000`
- **You should see**: PLC Logic Decompiler welcome page

---

## 🔧 **OPTION 2: Alternative Start Methods**

### **Method A: Direct Python Execution**
```cmd
# Activate environment first
conda activate pyoccenv

# Navigate to project
cd "c:\Users\jjacominiprioli\OneDrive - North Carolina A&T State University\Articles\PLC logic decompiler"

# Run application
python app.py
```

### **Method B: Using Conda Run**
```cmd
# Navigate to project
cd "c:\Users\jjacominiprioli\OneDrive - North Carolina A&T State University\Articles\PLC logic decompiler"

# Run with conda
conda run -n pyoccenv python app.py
```

---

## 🌐 **ACCESSING THE APPLICATION**

### **Default URLs:**
- **Main Application**: `http://127.0.0.1:5000`
- **Upload Page**: `http://127.0.0.1:5000/upload`
- **API Health Check**: `http://127.0.0.1:5000/api/health`

### **Expected Output When Starting:**
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
 * Restarting with stat
 * Debugger is active!
```

---

## 📱 **USING THE WEB APPLICATION**

### **1. Homepage Features**
- **Project Overview** with feature descriptions
- **Upload Interface** for L5X files
- **Analysis Dashboard** for results
- **Code Generation** with AI integration
- **Advanced Features** for power users

### **2. Upload and Process L5X Files**
1. **Click "Upload L5X File"** or navigate to `/upload`
2. **Select your L5X file** (like `Assembly_Controls_Robot.L5X`)
3. **Click "Upload and Process"**
4. **Wait for processing** (background processing with progress)
5. **View results** on analysis dashboard

### **3. Generate Python Code**
1. **After processing L5X**, click "Generate Code"
2. **Select generation type**:
   - Basic Interface
   - Full Interface (Recommended)
   - Safety Monitor
   - Data Logger
3. **Choose quality level**:
   - Basic
   - Production (Recommended)
   - Enterprise
4. **Select framework**: pycomm3 (default)
5. **Click "Generate"** and wait for AI processing
6. **Download generated code**

### **4. Advanced Features**
- **Context-aware code generation**
- **Multi-model AI coordination**
- **Learning and adaptation**
- **Pattern recognition**
- **Code validation and optimization**

---

## 🛠️ **TROUBLESHOOTING**

### **Common Issues & Solutions**

#### **Issue 1: Import Errors**
```
ModuleNotFoundError: No module named 'flask'
```
**Solution:**
```cmd
conda run -n pyoccenv pip install flask werkzeug jinja2 itsdangerous click markupsafe
```

#### **Issue 2: Port Already in Use**
```
OSError: [WinError 10048] Only one usage of each socket address
```
**Solution:**
```cmd
# Find and kill process using port 5000
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F

# Or use different port
# Edit app.py line: app.run(host='0.0.0.0', port=5001, debug=True)
```

#### **Issue 3: File Path Issues**
```
FileNotFoundError: No such file or directory
```
**Solution:**
```cmd
# Ensure you're in the correct directory
cd "c:\Users\jjacominiprioli\OneDrive - North Carolina A&T State University\Articles\PLC logic decompiler"

# Verify app.py exists
dir app.py
```

#### **Issue 4: Permission Errors**
```
PermissionError: [WinError 32] The process cannot access the file
```
**Solution:**
```cmd
# Run Command Prompt as Administrator
# Or check file permissions and antivirus software
```

---

## ⚡ **PERFORMANCE OPTIMIZATION**

### **For Better Performance:**

#### **1. Increase Memory (if needed)**
```cmd
# Set environment variables before starting
set FLASK_ENV=production
set PYTHONHASHSEED=0
python app.py
```

#### **2. Use Production Mode**
```cmd
# Edit app.py and change debug=False for production
# Or set environment variable
set FLASK_DEBUG=0
python app.py
```

#### **3. Clear Cache (if issues)**
```cmd
# Clear Python cache
rmdir /s __pycache__
rmdir /s src\__pycache__

# Clear browser cache and cookies
```

---

## 🔒 **SECURITY CONSIDERATIONS**

### **For Local Development:**
- ✅ **Default settings are safe** for local use
- ✅ **Debug mode enabled** for development
- ✅ **File upload restrictions** (16MB max, L5X files only)
- ✅ **Session security** with secret key

### **For Production Deployment:**
- 🔧 **Disable debug mode** (`debug=False`)
- 🔧 **Use production WSGI server** (Gunicorn, uWSGI)
- 🔧 **Set up reverse proxy** (Nginx, Apache)
- 🔧 **Configure HTTPS** with SSL certificates
- 🔧 **Set secure secret key** in environment variables

---

## 📊 **WHAT YOU CAN DO WITH THE APPLICATION**

### **✅ Core Features Available:**
1. **L5X File Processing**: Upload and parse Rockwell PLC files
2. **Tag Analysis**: Extract and analyze controller, program, and I/O tags
3. **Logic Analysis**: Interpret ladder logic and instruction patterns
4. **Knowledge Graph**: Visualize PLC system relationships
5. **AI Code Generation**: Generate Python interfaces using advanced AI
6. **Multi-Quality Levels**: Basic to enterprise-grade code generation
7. **Validation Systems**: Comprehensive code validation and optimization
8. **Interactive Dashboard**: Real-time analysis and results display
9. **Export Capabilities**: Download generated code and reports
10. **Advanced AI Features**: Context-aware generation with learning

### **📁 Supported File Types:**
- ✅ **L5X files** (Rockwell Automation/Allen-Bradley)
- ✅ **XML format** PLC programs
- ✅ **Maximum file size**: 16MB

### **🎯 Generated Output:**
- ✅ **Python code** for PLC communication (pycomm3)
- ✅ **Interface classes** for tag read/write operations
- ✅ **Documentation** and code comments
- ✅ **Error handling** and validation
- ✅ **Safety considerations** for industrial use

---

## 🎉 **SUCCESS CONFIRMATION**

### **You'll Know It's Working When:**
1. ✅ **Terminal shows**: `Running on http://127.0.0.1:5000`
2. ✅ **Browser loads**: PLC Logic Decompiler homepage
3. ✅ **Upload works**: You can upload L5X files successfully
4. ✅ **Processing completes**: Analysis results appear
5. ✅ **Code generation works**: AI generates Python code
6. ✅ **Downloads work**: You can download generated files

### **Test with Sample File:**
- ✅ **Use**: `Assembly_Controls_Robot.L5X` (already in your project)
- ✅ **Expected**: Successful parsing with 100+ tags detected
- ✅ **Features**: Timer analysis, I/O mapping, code generation

---

## 🆘 **NEED HELP?**

### **If you encounter issues:**
1. **Check terminal output** for error messages
2. **Verify conda environment** is activated
3. **Ensure all dependencies** are installed
4. **Check file paths** are correct
5. **Try restarting** the application
6. **Clear browser cache** if web interface issues

### **Getting Support:**
- **Check project documentation** in `/docs` folder
- **Review error logs** in terminal output
- **Test with sample files** first
- **Use debug mode** to see detailed error information

---

## 🚀 **READY TO GO!**

**Your PLC Logic Decompiler web application is ready for local deployment!**

**Quick Start Command:**
```cmd
cd "c:\Users\jjacominiprioli\OneDrive - North Carolina A&T State University\Articles\PLC logic decompiler" && conda activate pyoccenv && python app.py
```

**Then open**: `http://127.0.0.1:5000` in your browser

**Happy PLC analyzing! 🎊**
