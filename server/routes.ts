import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import multer from "multer";
import csvParser from "csv-parser";
import { Readable } from "stream";
import { z } from "zod";
import path from "path";
import { csvRowSchema, type CsvRow } from "@shared/schema";
import cors from "cors";
import session from "express-session";
import MemoryStore from "memorystore";

// Define memory store for sessions
const SessionStore = MemoryStore(session);

export async function registerRoutes(app: Express): Promise<Server> {
  // Enable CORS with specific configuration
  app.use(cors({
    origin: process.env.NODE_ENV === 'production' ? false : ['http://localhost:3000', 'http://localhost:5000'],
    credentials: true,
    allowedHeaders: ['Content-Type', 'Authorization'],
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    maxAge: 86400 // 24 hours
  }));

  // Setup session middleware
  app.use(session({
    secret: process.env.SESSION_SECRET || "csv-processor-secret",
    resave: false,
    saveUninitialized: false,
    cookie: { secure: process.env.NODE_ENV === "production", maxAge: 24 * 60 * 60 * 1000 },
    store: new SessionStore({
      checkPeriod: 86400000 // prune expired entries every 24h
    })
  }));

  // Configure multer for file uploads
  const upload = multer({
    storage: multer.memoryStorage(),
    limits: {
      fileSize: 5 * 1024 * 1024, // 5MB limit
    },
    fileFilter: (_req, file, cb) => {
      const ext = path.extname(file.originalname).toLowerCase();
      if (ext !== '.csv') {
        return cb(new Error('Only CSV files are allowed'));
      }
      cb(null, true);
    },
  });

  // Authentication middleware
  const authenticate = (req: Request, res: Response, next: Function) => {
    if (req.session.userId) {
      next();
    } else {
      res.status(401).json({ message: "Authentication required" });
    }
  };

  // Authentication routes
  app.post('/api/auth/login', async (req: Request, res: Response) => {
    try {
      const { username, password } = req.body;
      
      if (!username || !password) {
        return res.status(400).json({ message: "Username and password are required" });
      }
      
      const user = await storage.getUserByUsername(username);
      
      if (!user || user.password !== password) {
        return res.status(401).json({ message: "Invalid credentials" });
      }
      
      // Set user in session
      req.session.userId = user.id;
      
      res.status(200).json({ 
        message: "Login successful",
        user: { id: user.id, username: user.username }
      });
    } catch (error) {
      console.error("Login error:", error);
      res.status(500).json({ message: "Server error" });
    }
  });

  app.get('/api/auth/status', (req: Request, res: Response) => {
    if (req.session.userId) {
      res.status(200).json({ 
        authenticated: true,
        userId: req.session.userId
      });
    } else {
      res.status(200).json({ authenticated: false });
    }
  });

  app.post('/api/auth/logout', (req: Request, res: Response) => {
    req.session.destroy((err) => {
      if (err) {
        return res.status(500).json({ message: "Error logging out" });
      }
      res.status(200).json({ message: "Logout successful" });
    });
  });

  // CSV file upload route
  app.post('/api/csv/upload', authenticate, upload.single('file'), async (req: Request, res: Response) => {
    try {
      if (!req.file) {
        return res.status(400).json({ message: "No file uploaded" });
      }

      const file = req.file;
      const userId = req.session.userId as number;
      
      // Create CSV file record
      const csvFile = await storage.createCsvFile({
        fileName: file.originalname,
        fileSize: `${file.size}`,
        status: "processing",
        userId
      });

      // Parse CSV data
      const results: CsvRow[] = [];
      const stream = Readable.from(file.buffer);
      
      const processStream = new Promise<void>((resolve, reject) => {
        stream.pipe(csvParser())
          .on('data', (data) => {
            try {
              console.log("Processing CSV row:", data);
              
              // Handle different CSV formats by detecting fields
              let row: any;

              // Check for customer churn data format
              if (data.CustomerID !== undefined || data.Churn !== undefined) {
                // Customer churn data format
                row = {
                  id: data.CustomerID || '',
                  name: `${data.Gender || ''} (Age: ${data.Age || ''})`,
                  email: `customer${data.CustomerID || ''}@example.com`,
                  status: data.Churn === 'Yes' ? 'churned' : 'active',
                  lastUpdated: data.TotalCharges ? `Charges: $${data.TotalCharges}` : new Date().toISOString().split('T')[0]
                };
              } 
              // Handle header row
              else if (data.id === 'id' || data.ID === 'ID' || data.CustomerID === 'CustomerID') {
                // Skip header row
                return;
              }
              // Default format
              else {
                row = {
                  id: data.id || data.ID || data.Id || String(Math.floor(Math.random() * 10000)), // Fallback ID
                  name: data.name || data.Name || data.NAME || 'Unknown',
                  email: data.email || data.Email || data.EMAIL || 'unknown@example.com',
                  status: data.status || data.Status || data.STATUS || 'unknown',
                  lastUpdated: data.lastUpdated || data.LastUpdated || data['Last Updated'] || new Date().toISOString().split('T')[0]
                };
              }
              
              // Validate the row data
              const validatedRow = csvRowSchema.parse(row);
              results.push(validatedRow);
            } catch (error) {
              console.error("Invalid row data:", error);
              // Continue processing valid rows
            }
          })
          .on('end', async () => {
            try {
              // Store the processed CSV rows
              await storage.storeCsvRows(csvFile.id, results);
              
              // Update file status
              await storage.updateCsvFileStatus(csvFile.id, "processed");
              
              resolve();
            } catch (error) {
              reject(error);
            }
          })
          .on('error', (error) => {
            reject(error);
          });
      });
      
      await processStream;
      
      res.status(200).json({ 
        message: "File processed successfully", 
        fileId: csvFile.id,
        rowCount: results.length
      });
    } catch (error: any) {
      console.error("CSV upload error:", error);
      res.status(500).json({ message: `Error processing file: ${error.message}` });
    }
  });

  // Get recent CSV files
  app.get('/api/csv/recent', authenticate, async (req: Request, res: Response) => {
    try {
      const userId = req.session.userId as number;
      const files = await storage.getCsvFilesByUserId(userId);
      
      // If no files exist yet, return an empty array instead of an error
      res.status(200).json({
        files: files.length > 0 
          ? files.sort((a, b) => b.uploadedAt.getTime() - a.uploadedAt.getTime()).slice(0, 5)
          : []
      });
    } catch (error) {
      console.error("Error fetching recent files:", error);
      res.status(500).json({ message: "Server error" });
    }
  });

  // Get processed CSV data
  app.get('/api/csv/:fileId', authenticate, async (req: Request, res: Response) => {
    try {
      const fileId = parseInt(req.params.fileId);
      
      if (isNaN(fileId)) {
        return res.status(400).json({ message: "Invalid file ID" });
      }
      
      const file = await storage.getCsvFile(fileId);
      
      if (!file) {
        return res.status(404).json({ message: "File not found" });
      }
      
      // Validate user has access to this file
      if (file.userId !== req.session.userId) {
        return res.status(403).json({ message: "You don't have permission to access this file" });
      }
      
      const data = await storage.getCsvRowsByFileId(fileId);
      
      res.status(200).json({
        file,
        data
      });
    } catch (error) {
      console.error("Error fetching CSV data:", error);
      res.status(500).json({ message: "Server error" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
