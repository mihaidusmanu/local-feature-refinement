// Copyright (c) 2020, ETH Zurich, CVG, Mihai Dusmanu (mihai.dusmanu@inf.ethz.ch)

#include <Eigen/Core>

#include "ceres/ceres.h"

class BiquadraticInterpolator {
    public:
        explicit BiquadraticInterpolator(
            const std::vector<double>& data, size_t n_channels
        ) : data_(data), n_channels_(n_channels) {}

        void Evaluate(double row, double col, double* f, double* dfdrow, double* dfdcol) const {
            // Clipping.
            const double row_ = row;
            const double col_ = col;
            row = std::max(std::min(row, row_end_), row_start_);
            col = std::max(std::min(col, col_end_), col_start_);
            
            const double lagrange_row[3] = {2. * row * (row - .5), (-4.) * (row - .5) * (row + .5), 2. * row * (row + 0.5)};
            const double deriv_lagrange_row[3] = {2. * row + 2. * (row - .5), (-4.) * (row - .5) + (-4.) * (row + .5), 2. * row + 2. * (row + 0.5)};
            const double lagrange_col[3] = {2. * col * (col - .5), (-4.) * (col - .5) * (col + .5), 2. * col * (col + 0.5)};
            const double deriv_lagrange_col[3] = {2. * col + 2. * (col - .5), (-4.) * (col - .5) + (-4.) * (col + .5), 2. * col + 2. * (col + 0.5)};

            for (size_t k = 0; k < n_channels_; ++k) {
                f[k] = 0.;
                if (dfdrow != NULL) {
                    dfdrow[k] = 0.;
                    dfdcol[k] = 0.;
                }

                for (size_t i = 0; i < n_rows_; ++i) {
                    for (size_t j = 0; j < n_cols_; ++j) {
                        double data = data_[n_channels_ * (i * n_cols_ + j) + k];
                        f[k] += lagrange_row[i] * lagrange_col[j] * data;

                        if (dfdrow != NULL) {
                            if (row_ == row) {
                                dfdrow[k] += deriv_lagrange_row[i] * lagrange_col[j] * data;
                            }
                            if (col_ == col) {
                                dfdcol[k] += lagrange_row[i] * deriv_lagrange_col[j] * data;
                            }
                        }
                    }
                }
            }
        }

        // The following two Evaluate overloads are needed for interfacing with automatic differentiation.
        // The first is for when a scalar evaluation is done, and the second one is for when Jets are used.
        void Evaluate(const double& r, const double& c, double* f) const {
            Evaluate(r, c, f, NULL, NULL);
        }

        template<typename JetT> void Evaluate(const JetT& x, const JetT& y, JetT* f) const {
            double frc[n_channels_], dfdr[n_channels_], dfdc[n_channels_];
            Evaluate(x.a, y.a, frc, dfdr, dfdc);
            for (size_t k = 0; k < n_channels_; ++k) {
                f[k].a = frc[k];
                f[k].v = dfdr[k] * x.v + dfdc[k] * y.v;
            }
        }

    private:
        const std::vector<double> data_;
        const size_t n_channels_;
        const double row_start_ = -.5; const double col_start_ = -.5;
        const double row_end_ = .5; const double col_end_ = .5;
        const double one_over_row_stride_ = 2.; const double one_over_col_stride_ = 2.;
        const size_t n_rows_ = 3; const size_t n_cols_ = 3;
};

template<typename Interpolator> struct InterpolatedCostFunctor {
    public:
        explicit InterpolatedCostFunctor(Interpolator&& interpolator) : interpolator_(std::move(interpolator)) {}

    template<typename T> bool operator()(const T* x1, const T* x2, T* residuals) const {
        const Eigen::Map<const Eigen::Matrix<T, 2, 1>> x1_vec(x1);
        const Eigen::Map<const Eigen::Matrix<T, 2, 1>> x2_vec(x2);

        Eigen::Matrix<T, 2, 1> disp;
        interpolator_.Evaluate(x1_vec[0], x1_vec[1], disp.data());

        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals_vec(residuals);

        residuals_vec = x2_vec - x1_vec - disp;

        return true;
    }

    static ceres::CostFunction* Create(Interpolator&& interpolator) {
        return new ceres::AutoDiffCostFunction<InterpolatedCostFunctor, 2, 2, 2>(new InterpolatedCostFunctor(std::move(interpolator)));
    }

    private:
        const Interpolator interpolator_;
};
