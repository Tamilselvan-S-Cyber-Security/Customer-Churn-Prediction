export default function Footer() {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="bg-white border-t border-neutral-200 mt-12">
      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <p className="text-center text-sm text-neutral-500">
          &copy; {currentYear} CSV Data Processor. All rights reserved.
        </p>
      </div>
    </footer>
  );
}
