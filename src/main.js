const { app, BrowserWindow } = require('electron');

// include the Node.js 'path' module at the top of your file
const path = require('node:path')


// modify your existing createWindow() function
const createWindow = () => {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: true,
      nodeIntegrationInWorker: true,
      contextIsolation: false // separate context btw internal logic and website in webContents (make 'require' work)
    }
  })

  win.loadFile('./src/main.html')
}

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  });
})
