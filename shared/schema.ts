import { pgTable, text, serial, integer, boolean, timestamp } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const csvFiles = pgTable("csv_files", {
  id: serial("id").primaryKey(),
  fileName: text("file_name").notNull(),
  fileSize: text("file_size").notNull(),
  status: text("status").notNull(),
  uploadedAt: timestamp("uploaded_at").notNull().defaultNow(),
  userId: integer("user_id").references(() => users.id),
});

export const csvData = pgTable("csv_data", {
  id: serial("id").primaryKey(),
  fileId: integer("file_id").references(() => csvFiles.id),
  rowData: text("row_data").notNull(),
  status: text("status").notNull(),
  lastUpdated: timestamp("last_updated").notNull().defaultNow(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export const insertCsvFileSchema = createInsertSchema(csvFiles).pick({
  fileName: true,
  fileSize: true,
  status: true,
  userId: true,
});

export const insertCsvDataSchema = createInsertSchema(csvData).pick({
  fileId: true,
  rowData: true,
  status: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

export type InsertCsvFile = z.infer<typeof insertCsvFileSchema>;
export type CsvFile = typeof csvFiles.$inferSelect;

export type InsertCsvData = z.infer<typeof insertCsvDataSchema>;
export type CsvData = typeof csvData.$inferSelect;

export const csvRowSchema = z.object({
  id: z.string().or(z.number().transform(val => String(val))),
  name: z.string().default(''),
  // Make email more flexible to handle various formats in CSV files
  email: z.string().email().or(z.string().default('unknown@example.com')),
  status: z.string().default('unknown'),
  lastUpdated: z.string().default(new Date().toISOString().split('T')[0]),
});

export type CsvRow = z.infer<typeof csvRowSchema>;
