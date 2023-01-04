const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const MiniCssExtractPlugin = require("mini-css-extract-plugin");
const BundleAnalyzerPlugin = require('webpack-bundle-analyzer').BundleAnalyzerPlugin;

const isProduction = process.env.NODE_ENV == "production";

const stylesHandler = isProduction
  ? MiniCssExtractPlugin.loader
  : "style-loader";

const config = {
  context: path.resolve(__dirname, "src"),
  entry: "./main.ts",
  output: {
    path: path.resolve(__dirname, "dist"),
  },
  plugins: [
    // Enable this plugin to launch a treemap Viz of bundle size
    // new BundleAnalyzerPlugin(),
    new HtmlWebpackPlugin({
      favicon: "./assets/favicon.png",
      template: "index.html",
    }),
  ],
  module: {
    rules: [
      {
        test: /\.(ts|tsx)$/i,
        use: ["ts-loader"],
        exclude: ["/node_modules/"],
      },
      {
        test: /\.css$/i,
        use: [stylesHandler, "css-loader"],
      },
      {
        test: /\.(gif|jpg|jpeg|png|svg)$/i,
        type: "asset/resource",
      },
    ],
  },
  performance: {
    maxEntrypointSize: 512000,
    maxAssetSize: 512000
  },
  resolve: {
    extensions: [".tsx", ".ts", ".jsx", ".js", "..."],
  },
  devServer: {
    static: {
      directory: path.join(__dirname, 'dist'),
    },
    port: 3000,
    open: true,
    host: "localhost",
    hot: true,
    watchFiles: ["./src/*"]
  },
};

module.exports = () => {
  if (isProduction) {
    config.mode = "production";
    config.devtool = false

    config.plugins.push(new MiniCssExtractPlugin());
  } else {
    config.mode = "development";
    config.devtool = "eval"
  }
  return config;
};
