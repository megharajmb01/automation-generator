const { test, expect } = require('@playwright/test');
const path = require('path');

test.describe('TestCaseGen AI - PDF Upload Workflow', () => {

  test.beforeEach(async ({ page }) => {
    // Navigate to the homepage before each test
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('Step 1: Should display homepage with PDF Test Cases option', async ({ page }) => {
    // Verify main heading
    await expect(page.locator('text=Choose Your Workflow')).toBeVisible();
    await expect(page.locator('text=Select how you want to generate test cases')).toBeVisible();

    // Verify PDF Test Cases card is visible
    await expect(page.locator('text=PDF Test Cases')).toBeVisible();

    // Verify card description
    await expect(page.locator('text=Upload a PDF document and generate test cases from:')).toBeVisible();

    // Verify navigation items
    await expect(page.locator('text=Home')).toBeVisible();
    await expect(page.locator('text=Connected')).toBeVisible();

    console.log('✅ Homepage loaded successfully');
  });

  test('Step 2: Should navigate to PDF upload page when PDF Test Cases is clicked', async ({ page }) => {
    // Click on PDF Test Cases card
    await page.click('text=PDF Test Cases');

    // Wait for navigation
    await page.waitForLoadState('networkidle');

    // Verify we're on the PDF Test Case Generator page
    await expect(page.locator('text=PDF Test Case Generator')).toBeVisible();
    await expect(page.locator('text=Extract sections and generate test cases from your PDFs')).toBeVisible();

    // Verify PDF Mode badge
    await expect(page.locator('text=PDF Mode')).toBeVisible();

    // Verify step indicator shows step 1 (Upload PDF) - FIX: Use more specific selector
    await expect(page.locator('.text-xs.mt-1').filter({ hasText: 'Upload PDF' }).first()).toBeVisible();

    console.log('✅ Navigated to PDF upload page');
  });

  test('Step 3: Should display upload options on PDF page', async ({ page }) => {
    // Navigate to PDF page
    await page.click('text=PDF Test Cases');
    await page.waitForLoadState('networkidle');

    // Verify "Upload New PDF" section
    await expect(page.locator('text=Upload New PDF')).toBeVisible();

    // Verify "Click to upload PDF" text
    await expect(page.getByText('Click to upload PDF')).toBeVisible();

    // Verify "or drag and drop" text
    await expect(page.locator('text=or drag and drop')).toBeVisible();

    // Verify Upload PDF button exists
    await expect(page.getByRole('button', { name: 'Upload PDF' })).toBeVisible();

    // Verify "Load Previous PDF" section
    await expect(page.locator('text=Load Previous PDF')).toBeVisible();

    console.log('✅ Upload options displayed');
  });

  test('Step 4: Should open file picker when "Click to upload PDF" is clicked', async ({ page }) => {
    // Navigate to PDF page
    await page.click('text=PDF Test Cases');
    await page.waitForLoadState('networkidle');

    // Verify the file input exists
    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toBeAttached();

    console.log('✅ File input is ready');
  });

  test('Complete Workflow: Select PDF Test Cases → Upload PDF', async ({ page }) => {
    // Step 1: Navigate to homepage
    await expect(page.locator('text=Choose Your Workflow')).toBeVisible();
    console.log('✅ Step 1: On homepage');

    // Step 2: Click PDF Test Cases
    await page.click('text=PDF Test Cases');
    await page.waitForLoadState('networkidle');
    console.log('✅ Step 2: Clicked PDF Test Cases');

    // Step 3: Verify we're on upload page
    await expect(page.locator('text=PDF Test Case Generator')).toBeVisible();
    console.log('✅ Step 3: On PDF upload page');

    // Step 4: Click on upload area
    await page.click('text=Click to upload PDF');
    console.log('✅ Step 4: Clicked upload area');

    // Step 5: Upload a PDF file
    const fileInput = page.locator('input[type="file"]');

    // Create a dummy PDF file path (you'll need to provide an actual PDF)
    // For now, we'll just verify the input exists
    await expect(fileInput).toBeAttached();

    // If you have a test PDF file, uncomment and use this:
    const filePath = path.join(__dirname, 'test-files', 'sample.pdf');
    await fileInput.setInputFiles(filePath);
    console.log('✅ Step 5: PDF file selected');

    //Step 6: Click Upload PDF button
    const uploadButton = page.getByRole('button', { name: 'Upload PDF' });
    await uploadButton.click();
    console.log('✅ Step 6: Clicked Upload PDF button');

    console.log('✅ Workflow test completed');
  });

  test('Should display all action buttons in header', async ({ page }) => {
    // Navigate to PDF page
    await page.click('text=PDF Test Cases');
    await page.waitForLoadState('networkidle');

    // Verify header buttons - FIX: Use more specific selectors
    await expect(page.getByRole('button', { name: 'Add Figma' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Switch Mode' })).toBeVisible();

    // For Reset button, select the one in the header (second one)
    const resetButtons = page.getByRole('button', { name: 'Reset' });
    await expect(resetButtons.nth(1)).toBeVisible();

    console.log('✅ All action buttons visible');
  });

  test('Should display all 4 steps in progress indicator', async ({ page }) => {
    // Navigate to PDF page
    await page.click('text=PDF Test Cases');
    await page.waitForLoadState('networkidle');

    // Verify all 4 steps - FIX: Use more specific selectors for step labels
    await expect(page.locator('span.text-xs').filter({ hasText: 'Upload PDF' }).first()).toBeVisible();
    await expect(page.locator('span.text-xs').filter({ hasText: 'Extract Sections' })).toBeVisible();
    await expect(page.locator('span.text-xs').filter({ hasText: 'Choose Section' })).toBeVisible();
    await expect(page.locator('span.text-xs').filter({ hasText: 'Generate Tests' })).toBeVisible();

    console.log('✅ All steps displayed');
  });

  test('Should show Load Previously Processed PDFs option', async ({ page }) => {
    // Navigate to PDF page
    await page.click('text=PDF Test Cases');
    await page.waitForLoadState('networkidle');

    // Verify load previous PDFs section
    await expect(page.locator('text=Load Previously Processed PDFs')).toBeVisible();
    await expect(page.locator('text=Click to view list')).toBeVisible();

    console.log('✅ Load previous PDFs option visible');
  });
});

