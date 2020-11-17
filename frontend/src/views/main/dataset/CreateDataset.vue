<template>
  <v-container fluid>
    <v-card class="ma-3 pa-3">
      <v-card-title primary-title>
        <div class="headline primary--text">Create Dataset</div>
      </v-card-title>
      <v-card-text>
        <template>
          <v-form v-model="valid" ref="form" lazy-validation>
            <v-text-field label="Title" v-model="new_dataset.title" required></v-text-field>
            <v-text-field label="Description" v-model="new_dataset.description"></v-text-field>
            <v-select label="File type" v-model="new_dataset.file_type" :items="file_types"></v-select>
            <v-text-field label="Location" v-model="new_dataset.location"></v-text-field>
            <v-text-field label="Resolution" v-model="new_dataset.resolution"></v-text-field>
            <v-text-field label="Modality" v-model="new_dataset.modality"></v-text-field>
            <!-- <v-text-field label="Full Name" v-model="title" required></v-text-field>
            <v-text-field label="E-mail" type="description" v-model="description" v-validate="'required|description'" data-vv-name="description" :error-messages="errors.collect('description')" required></v-text-field>
            <div class="subheading secondary--text text--lighten-2">Dataset is superdataset <span v-if="isSuperdataset">(currently is a superdataset)</span><span v-else>(currently is not a superdataset)</span></div>
            <v-checkbox label="Is Superdataset" v-model="isSuperdataset"></v-checkbox>
            <div class="subheading secondary--text text--lighten-2">Dataset is active <span v-if="isActive">(currently active)</span><span v-else>(currently not active)</span></div>
            <v-checkbox label="Is Active" v-model="isActive"></v-checkbox>
            <v-layout align-center>
              <v-flex>
                <v-text-field type="password" ref="password" label="Set Password" data-vv-name="password" data-vv-delay="100" v-validate="{required: true}" v-model="password1" :error-messages="errors.first('password')">
                </v-text-field>
                <v-text-field type="password" label="Confirm Password" data-vv-name="password_confirmation" data-vv-delay="100" data-vv-as="password" v-validate="{required: true, confirmed: 'password'}" v-model="password2" :error-messages="errors.first('password_confirmation')">
                </v-text-field>
              </v-flex>
            </v-layout> -->
          </v-form>
        </template>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn @click="cancel">Cancel</v-btn>
        <v-btn @click="reset">Reset</v-btn>
        <v-btn @click="submit" :disabled="!valid">
              Save
            </v-btn>
      </v-card-actions>
    </v-card>
  </v-container>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import {
  Dataset,
  DatasetUpdate,
  DatasetCreate,
  DatasetFileType,
} from '@/api';
import {
  defaultDataset,
} from '@/interfaces';
import { dispatchGetDatasets, dispatchCreateDataset } from '@/store/dataset/actions';

function filterUndefined(obj) {
  Object.keys(obj).forEach(key => obj[key] === undefined && delete obj[key]);
  return obj;
}

@Component
export default class CreateDataset extends Vue {
  public new_dataset: DatasetCreate = defaultDataset();
  public file_types = Object.values(DatasetFileType);

  public async mounted() {
    await dispatchGetDatasets(this.$store);
    this.reset();
  }

  public reset() {
    this.new_dataset = defaultDataset();
    this.$validator.reset();
  }

  public cancel() {
    this.$router.back();
  }

  public async submit() {
    if (await this.$validator.validateAll()) {
      var filteredDataset : DatasetCreate = filterUndefined(this.new_dataset);
      console.log(filteredDataset);
      await dispatchCreateDataset(this.$store, filteredDataset);
      this.$router.push('/main/datasets');
    }
  }
}
</script>
