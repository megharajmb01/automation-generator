
const { test, expect } = require('@playwright/test');

test.describe('AI Generated Playwright Tests', () => {

  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
    await page.getByRole('button', { name: 'PDF Test Cases' }).click();
  });

test('AC-TC-001 Home page loads correctly', async ({ page }) => {
  await expect(page.getByText('Four Step Automation')).toBeVisible();
  await expect(page.locator('.step-item').filter({ hasText: 'Upload PDF' })).toBeVisible();
  await expect(page.getByText('Extract Sections')).toBeVisible();
  await expect(page.getByText('Choose Section')).toBeVisible();
  await expect(page.getByText('Generate Tests')).toBeVisible();
  await expect(page.locator('div[class*="upload-section"]', {hasText: 'Click to upload or drag and drop'})).toBeVisible();
  await expect(page.getByRole('button', { name: 'Click to view list' })).toBeVisible();
});
test('AC-TC-002 Verify successful PDF upload using the \'Click to upload\' method', async ({ page }) => {
  await page.locator('input[type="file"]').setInputFiles('/home/megharaj/Automation Generator/frontend/tests/test-files/sample.pdf');
  await page.getByRole('button', { name: 'Upload PDF' }).click();
  await expect(page.getByText('Ready to Extract')).toBeVisible();
});
test( 'AC-TC-003 Verify successful PDF upload using the drag-and-drop method', async ({ page }) => {
  await page.locator('input[type="file"]').setInputFiles('/home/megharaj/Automation Generator/frontend/tests/test-files/sample.pdf');
  await page.getByRole('button', { name: 'Upload PDF' }).click();
  await expect(page.getByText('Ready to Extract')).toBeVisible();
});

});
