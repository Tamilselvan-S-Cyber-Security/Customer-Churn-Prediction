import { 
  users, type User, type InsertUser,
  csvFiles, type CsvFile, type InsertCsvFile,
  csvData, type CsvData, type InsertCsvData,
  type CsvRow
} from "@shared/schema";

export interface IStorage {
  // User operations
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  
  // CSV file operations
  getCsvFile(id: number): Promise<CsvFile | undefined>;
  getCsvFilesByUserId(userId: number): Promise<CsvFile[]>;
  createCsvFile(file: InsertCsvFile): Promise<CsvFile>;
  updateCsvFileStatus(id: number, status: string): Promise<CsvFile | undefined>;
  getRecentCsvFiles(limit: number): Promise<CsvFile[]>;
  
  // CSV data operations
  getCsvDataByFileId(fileId: number): Promise<CsvData[]>;
  createCsvData(data: InsertCsvData): Promise<CsvData>;
  storeCsvRows(fileId: number, rows: CsvRow[]): Promise<void>;
  getCsvRowsByFileId(fileId: number): Promise<CsvRow[]>;
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private csvFiles: Map<number, CsvFile>;
  private csvData: Map<number, CsvData>;
  private csvRows: Map<number, CsvRow[]>;
  private userId: number;
  private fileId: number;
  private dataId: number;

  constructor() {
    this.users = new Map();
    this.csvFiles = new Map();
    this.csvData = new Map();
    this.csvRows = new Map();
    this.userId = 1;
    this.fileId = 1;
    this.dataId = 1;
    
    // Initialize with a default user
    const defaultUser: User = {
      id: this.userId,
      username: "user@example.com",
      password: "password123"
    };
    this.users.set(defaultUser.id, defaultUser);
  }

  // User operations
  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = this.userId++;
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }

  // CSV file operations
  async getCsvFile(id: number): Promise<CsvFile | undefined> {
    return this.csvFiles.get(id);
  }

  async getCsvFilesByUserId(userId: number): Promise<CsvFile[]> {
    return Array.from(this.csvFiles.values()).filter(
      (file) => file.userId === userId,
    );
  }

  async createCsvFile(insertFile: InsertCsvFile): Promise<CsvFile> {
    const id = this.fileId++;
    const now = new Date();
    const file: CsvFile = { 
      ...insertFile, 
      id, 
      uploadedAt: now
    };
    this.csvFiles.set(id, file);
    return file;
  }

  async updateCsvFileStatus(id: number, status: string): Promise<CsvFile | undefined> {
    const file = this.csvFiles.get(id);
    if (!file) return undefined;
    
    const updatedFile = { ...file, status };
    this.csvFiles.set(id, updatedFile);
    return updatedFile;
  }

  async getRecentCsvFiles(limit: number): Promise<CsvFile[]> {
    return Array.from(this.csvFiles.values())
      .sort((a, b) => b.uploadedAt.getTime() - a.uploadedAt.getTime())
      .slice(0, limit);
  }

  // CSV data operations
  async getCsvDataByFileId(fileId: number): Promise<CsvData[]> {
    return Array.from(this.csvData.values()).filter(
      (data) => data.fileId === fileId,
    );
  }

  async createCsvData(insertData: InsertCsvData): Promise<CsvData> {
    const id = this.dataId++;
    const now = new Date();
    const data: CsvData = {
      ...insertData,
      id,
      lastUpdated: now
    };
    this.csvData.set(id, data);
    return data;
  }

  async storeCsvRows(fileId: number, rows: CsvRow[]): Promise<void> {
    this.csvRows.set(fileId, rows);
  }

  async getCsvRowsByFileId(fileId: number): Promise<CsvRow[]> {
    return this.csvRows.get(fileId) || [];
  }
}

export const storage = new MemStorage();
